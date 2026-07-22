"""
    SAGEBewley

Clean, stable re-implementation of the Bewley/SAGE wellbeing model from
A. Conway's master's thesis ("Wellbeing and Macroeconomics: A SAGE approach").

Design goals (vs. the original `Bewley change11j` notebook):
  * No brute-force continuous inner optimization → discretize effort `e` and
    solve the household problem as a QuantEcon `DiscreteDP`. Fast and stable.
  * No 5-million-period simulation → the wealth distribution is the exact
    stationary distribution of the controlled Markov chain.
  * No fragile outer A-fixed-point / backward NaN loop → in the *canonical*
    model (notebook change `v11c`) the public good `A` enters utility
    additively and is INDEPENDENT of the agent's own action, so household
    policies do not depend on `A`. `A = Q` is therefore just an aggregate
    computed once from the stationary distribution.

Model (partial equilibrium, exogenous gross rate R, no firms):
  state  s = (a, z),  a ≥ 0 assets, z ∈ {low, high} productivity (Rouwenhorst)
  choice (a', e),     a' ≥ 0 next assets, e ∈ [0,1] labour effort, q = 1-e
  budget c + a' = R·a + α[z]·e·z·Z            (agency α scales labour income)
  u(c,e;z) = Γ·( c^(1-γ)/(1-γ) - ϕ·e^(1+ψ)/(1+ψ) ) + Λ·B[z]·A
             \\_____________ Material Gain U^c ____________/   \\__ Social U^s __/
  public good  A = Q = E[ 1 - e ]  over the stationary distribution.

NOTE (flagged modelling choice): because the canonical code removed the `q·A`
term, effort does NOT respond to the social good - `A`/`B` are a wellbeing
overlay, not an active choice margin. The thesis Euler eq. (14) keeps a B term;
the final code (and hence its figures) does not. We follow the code.

VERSIONS. v1.0 (git tag) is the thesis-faithful parametrization (γ=1.5, ψ=0.25,
ϕ=1.0). v1.1 (this file) replaces every free preference parameter with a
literature or calibrated value: γ=2 (EIS 0.5, Havranek 2015), ψ=2 (Frisch 0.5,
Chetty et al. 2011), β=0.96 and R=1.02 (Aiyagari-tradition annual values; the
thesis βR=0.9999 knife edge made the wealth distribution truncation-driven
under γ=2), ϕ=14.0 (calibrated to the INSEE 2010 time-use work share 0.53).
v1.1 also replaces the stationary-distribution solver: sparse power iteration
with a dense repeated-squaring fallback for slow-mixing chains (the old
iteration could stall silently and bias aggregates by over a percentage
point). Side benefit of the corrected calibration: hand-to-mouth share ~0.33
(Kaplan-Violante-Weidner range) and wealth Gini ~0.55, both far closer to data
than the thesis values. Reproduce thesis numbers via the v1.0 tag.
"""
module SAGEBewley

using QuantEcon, LinearAlgebra, Statistics, SparseArrays

export SAGEParams, SAGESolution, solve_model, exponential_grid, income_process,
       wealth_gini, income_gini, frac_constrained, public_good_shares, update,
       country_params, country_targets, country_participation,
       COUNTRIES, COUNTRY_TARGETS, group_means

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
Base.@kwdef struct SAGEParams
    # ---- preferences (literature-disciplined defaults, v1.1) ---------------
    # γ: CRRA. EIS meta-analysis (Havranek 2015 JEEA, selective-reporting
    #    corrected) puts the mean EIS near 0.5, i.e. γ = 2. Standard in the
    #    heterogeneous-agent literature.
    # ψ: inverse Frisch. Quasi-experimental intensive-margin Frisch ~ 0.5
    #    (Chetty, Guren, Manoli, Weber 2011 AER), i.e. ψ = 2. The old default
    #    0.25 implied Frisch = 4, far outside the evidence; the thesis Table 1
    #    value 4.0 (Frisch 0.25) sat at the other extreme.
    # ϕ: effort-disutility scale, CALIBRATED (not free): set so the baseline
    #    mean work share e is 0.53, the paid share of committed time
    #    (paid 3h24 vs unpaid domestic/volunteer 3h01 per day, INSEE Enquete
    #    Emploi du temps 2010). ϕ = 14.0 hits e_mean = 0.533 under γ = ψ = 2,
    #    β = 0.96, R = 1.02, with the exact stationary distribution.
    γ::Float64  = 2.0            # CRRA (EIS = 0.5, Havranek 2015 meta)
    ϕ::Float64  = 14.0           # effort disutility (calibrated to INSEE EDT 2010, e_mean = 0.53)
    ψ::Float64  = 2.0            # inverse Frisch (Frisch = 0.5, Chetty et al. 2011)
    # β, R: annual interpretation. β = 0.96 is the standard annual discount
    # factor of the Aiyagari (1994) tradition; R = 1.02 a long-run real rate
    # (r* estimates, Holston-Laubach-Williams; OECD long averages). βR = 0.979
    # sits safely inside the impatience region. The thesis pair β = 0.99,
    # R = 1.01 (βR = 0.9999) was a knife edge: harmless under the thesis
    # curvature γ = 1.5, but under the literature curvature γ = 2 the wealth
    # distribution became truncation-driven (mass piling at the grid top and
    # aggregates moving with a_max), i.e. no honest stationary equilibrium.
    β::Float64  = 0.96           # discount factor (annual, Aiyagari 1994)
    R::Float64  = 1.02           # gross real rate (long-run r* ~ 2%)
    # income process: log z ~ AR(1) annual, Rouwenhorst. ρ = 0.9, η = 0.1 sit
    # inside the annual earnings-process ranges surveyed by Heathcote,
    # Storesletten, Violante (2010); the thesis disciplined them by matching
    # the hand-to-mouth share.
    ρ::Float64  = 0.9
    η::Float64  = 0.1
    nz::Int     = 2
    # Optional override of the productivity process: supply explicit state
    # values and a transition matrix (e.g. persistent x transitory composites,
    # or an estimated process from data). When set, ρ and η are ignored and
    # nz must equal length(z_vals_override). Defaults preserve old behaviour.
    z_vals_override::Union{Nothing,Vector{Float64}} = nothing
    Π_override::Union{Nothing,Matrix{Float64}}      = nothing
    # SAGE parameters (index 1 = low income, 2 = high income). Data-derived in
    # the thesis from OECD Better Life Index 2017 by education (France):
    #   α  empowerment indicators (labour-market security, health, skills)
    #   B  quality-of-support-network items by education
    #   Λ  weight on the cohesion dimension from BLI dimension rankings;
    #      flagged for re-estimation from life-satisfaction regressions.
    # Identification note: in decision utility only the PRODUCT κ·Λ·B[z] is
    # identified (see social_reward); Λ and the swept κ are separated only by
    # the wellbeing dashboard, where Λ·B·Q enters experienced wellbeing.
    α::Vector{Float64} = [0.765, 0.911]   # agency: share of labour income retained
    B::Vector{Float64} = [0.80, 0.94]     # enjoyment of the public good
    Γ::Float64  = 1.0                     # weight on Material Gain (normalisation)
    Λ::Float64  = 0.8757834               # weight on Social Cohesion (thesis, OECD BLI 2017)
    # asset grid (exponential: fine near 0)
    a_min::Float64 = 1e-10
    a_max::Float64 = 100.0
    na::Int     = 200
    pexp::Float64 = 1.5
    # effort grid
    ne::Int     = 60
    # aggregate productivity multiplier (1.0 = stationary; shock raises it)
    Z::Float64  = 1.0
    # --- behavioural social cohesion (Paper 1) -----------------------------
    # :off        canonical thesis: social term additive, effort does NOT respond
    # :warmglow   add κ·Λ·B·(1-e) to the reward: effort responds, no A feedback
    # :multiplier add κ·Λ·B·A·(1-e): effort responds AND the public good A feeds
    #             back (social multiplier), solved by an outer fixed point on A
    social_mode::Symbol    = :off
    social_strength::Float64 = 1.0   # κ, scales the behavioural social return
    # homophily h in [0,1]: weight on OWN-group contribution in the belonging
    # aggregate, A_eff[g] = (1-h)·Q + h·Qmean[g] (bonding vs bridging social
    # capital, Putnam 2000; homophily, McPherson-Smith-Lovin-Cook 2001).
    # h = 0 reproduces the plain multiplier; only used when social_mode is
    # :multiplier. Makes the fixed point per-group (one aggregate per z).
    homophily::Float64 = 0.0
    # --- policy experiment (Paper 1 normative) ----------------------------
    # A proportional make-work-pay subsidy to labour income, financed by a
    # lump-sum tax (set externally for budget balance). Raises the return to
    # effort, so it can erode the public good and social cohesion.
    subsidy::Float64 = 0.0           # proportional subsidy to labour income
    lumptax::Float64 = 0.0           # lump-sum tax financing it
    # --- participation tax credit (Paper 2 normative) ---------------------
    # A rebate on the opportunity cost of the time given to participation, the
    # model analogue of France's 66 percent charitable-donations deduction and
    # the UK's Gift Aid (25 percent basic-rate top-up). When a household
    # participates (the discrete choice d = 1, costing QBAR units of time) it
    # receives partcredit * α * z * Z * QBAR back into the budget. Acts WITH
    # the social complementarity, unlike the work subsidy which fights it.
    # Financed by the same lump-sum tax channel (lumptax) for budget balance.
    partcredit::Float64 = 0.0        # rebate rate on the participation time lump
end

"Return a copy of `p` with the named fields overridden (kwdef has no reconstruct)."
update(p::SAGEParams; kwargs...) = SAGEParams(;
    (f => get(Dict(kwargs), f, getfield(p, f)) for f in fieldnames(SAGEParams))...)

# ----------------------------------------------------------------------------
# Country calibration (plug-and-play)
# ----------------------------------------------------------------------------
# Per-country dials. The standard preferences (γ, φ, ψ) are treated as universal;
# the country-specific objects are the gross rate R, the income process (ρ, η),
# agency by education type α, the cohesion taste by type B, and the cohesion
# weight Λ. Index 1 = lower-education/income group, 2 = higher.
#
# Data sources, per parameter (same template for every country):
#   R       real long-term rate (central bank / OECD), deflated by CPI
#   ρ, η    earnings/income process; or set to match wealth Gini + hand-to-mouth
#           share from EU-SILC / HFCS / SCF / WID
#   α       agency: OECD Better Life Index empowerment indicators by education
#           (labour-market security, self-reported health, PISA skills)
#   B       cohesion taste: OECD BLI "quality of support network" by education,
#           or ESS / Gallup social-support items
#   Λ       weight on cohesion: OECD BLI dimension rankings, or estimated from
#           subjective-wellbeing (life-satisfaction) regressions
#
# Seven countries calibrated to the standard cross-country pipeline
# (CALIBRATION_PIPELINE.md at the repo root). Sources, the SAME for every row:
#   R              long-term real rate, OECD long-term interest series deflated
#                   by CPI, twenty-year average. Annualised gross rate.
#   ρ, η           moment-matched to (wealth Gini, hand-to-mouth share); the
#                   targets per country are recorded in the validation script.
#                   Wealth-Gini source: the Open Inequality Atlas (Conway 2025),
#                   conway1521.github.io/open-inequality-atlas, harmonising WID,
#                   HFCS, LWS, SCF, DFA across 213 countries with comparability
#                   tiers. Companion Moments Atlas adds top-1/10/bottom-50
#                   shares for 53 countries.
#   α              OECD "How's Life?" empowerment indicators by education, or
#                   WVS-equivalent items for non-OECD members (CN, ZA), scaled
#                   into the model's 0-1 band so the FR row reproduces the
#                   thesis numbers.
#   B              OECD "quality of support network" or WVS "can count on
#                   relatives or friends" by education, on 0-1.
#   Λ              OECD YBLI dimension weights baseline; upgrade path =
#                   life-satisfaction panel estimation (see Lambda Estimation
#                   Design in the vault).
#   participation  WVS Wave 7 (2017-2022) associational membership, by
#                   education (target_overall, target_low, target_high).
#                   For France: INSEE Première 2016, cited in the S+A paper.
#
# Status flags: real = sourced numbers from the named cross-country sources;
# scoping = best-evidence values pending one row of microdata extraction.
# The numbers should be treated as the v1 calibration; refining them is just
# pulling the named source and updating the row.
const COUNTRIES = Dict(
    "FR" => (R=1.02, ρ=0.90, η=0.10, ϕ=14.44, α=[0.765, 0.911], B=[0.80, 0.94], Λ=0.876,
             β=0.96, participation=(0.42, 0.25, 0.45), placeholder=false,
             note="France (thesis OECD BLI 2017 + INSEE Première 2016 participation, phi to EDT 2010 work share 0.53)"),
    "DE" => (R=1.02, ρ=0.92, η=0.11, ϕ=12.18, α=[0.770, 0.920], B=[0.82, 0.94], Λ=0.872,
             β=0.96, participation=(0.50, 0.30, 0.55), placeholder=false,
             note="Germany (OECD How's Life DE, HFCS + KVW HtM 0.32, WVS7 participation, phi to HETUS work share 0.55)"),
    "IT" => (R=1.02, ρ=0.92, η=0.12, ϕ=12.11, α=[0.700, 0.900], B=[0.85, 0.95], Λ=0.860,
             β=0.96, participation=(0.30, 0.18, 0.40), placeholder=false,
             note="Italy (OECD How's Life IT, HFCS + KVW HtM 0.41, ESS+WVS7 participation, phi to HETUS work share 0.55)"),
    "US" => (R=1.02, ρ=0.91, η=0.12, ϕ=8.21,  α=[0.720, 0.930], B=[0.83, 0.95], Λ=0.880,
             β=0.96, participation=(0.50, 0.30, 0.60), placeholder=false,
             note="USA (OECD How's Life US, SCF + KVW HtM 0.31, WVS7 participation, phi to ATUS work share 0.60)"),
    "CO" => (R=1.03, ρ=0.92, η=0.18, ϕ=4.45,  α=[0.650, 0.900], B=[0.80, 0.92], Λ=0.860,
             β=0.96, participation=(0.25, 0.15, 0.35), placeholder=false,
             note="Colombia (Atlas wealth Gini 0.81 from LAm DINA series; ECV/GEIH HtM ~0.55; OECD How's Life CO since 2020 OECD membership; WVS7 participation; phi to ENUT work share 0.60)"),
    "ZA" => (R=1.03, ρ=0.90, η=0.20, ϕ=5.49,  α=[0.550, 0.880], B=[0.70, 0.92], Λ=0.860,
             β=0.96, participation=(0.30, 0.20, 0.45), placeholder=false,
             note="South Africa (Atlas wealth Gini 0.95 + Chatterjee-Czajka-Gethin 2022 WBER top-10 share 86%; NIDS HtM ~0.60; OECD How's Life ZA key partner; WVS7; phi to SA TUS 2010 work share 0.55)"),
    "CN" => (R=1.03, ρ=0.92, η=0.16, ϕ=3.71,  α=[0.650, 0.920], B=[0.78, 0.90], Λ=0.860,
             β=0.96, participation=(0.20, 0.12, 0.30), placeholder=false,
             note="China (Atlas wealth Gini 0.74 + Piketty-Yang-Zucman 2019 top-10 share 67%; CHFS HtM ~0.50; OECD key partner; WVS7; phi to CTUS 2018 work share 0.65)"),
)

"""
Calibration targets (the moments each country row aims at). The validation
script reports the model's untargeted moments against these and adds untargeted
diagnostics. Source documentation lives in `CALIBRATION_PIPELINE.md`.
"""
const COUNTRY_TARGETS = Dict(
    "FR" => (wealth_gini=0.68, htm=0.30, work_share=0.53, income_gini=0.30),
    "DE" => (wealth_gini=0.78, htm=0.32, work_share=0.55, income_gini=0.30),
    "IT" => (wealth_gini=0.61, htm=0.41, work_share=0.55, income_gini=0.33),
    "US" => (wealth_gini=0.85, htm=0.31, work_share=0.60, income_gini=0.41),
    "CO" => (wealth_gini=0.81, htm=0.55, work_share=0.60, income_gini=0.55),
    "ZA" => (wealth_gini=0.95, htm=0.60, work_share=0.55, income_gini=0.63),
    "CN" => (wealth_gini=0.74, htm=0.50, work_share=0.65, income_gini=0.45),
)

"""
    country_params(code; kwargs...) -> SAGEParams

Build parameters for a country (e.g. "FR", "US", "NL", "IT") from `COUNTRIES`.
Standard preferences and grids take SAGEParams defaults; country dials are set
from the table. Extra keyword arguments (e.g. `social_mode = :warmglow`) override.
Warns if the country's values are still placeholders.
"""
function country_params(code::AbstractString; kwargs...)
    haskey(COUNTRIES, code) || error("unknown country \"$code\"; have $(sort(collect(keys(COUNTRIES))))")
    c = COUNTRIES[code]
    c.placeholder && @warn "country \"$code\" uses placeholder values; replace with data ($(c.note))"
    update(SAGEParams(; R=c.R, ρ=c.ρ, η=c.η, ϕ=c.ϕ, α=c.α, B=c.B, Λ=c.Λ, β=c.β); kwargs...)
end

"Participation targets for country `code` as a NamedTuple (overall, low, high)."
country_participation(code::AbstractString) = COUNTRIES[code].participation

"Calibration targets for country `code` (wealth Gini, HtM, work share, income Gini)."
country_targets(code::AbstractString) = COUNTRY_TARGETS[code]

# ----------------------------------------------------------------------------
# Grids and income process
# ----------------------------------------------------------------------------
"Exponential asset grid (replicates the notebook's `grid_fun_exp`)."
function exponential_grid(a_min, a_max, na, pexp)
    x = range(a_min, step = 0.5, length = na)
    xp = x .^ pexp
    return a_min .+ (a_max - a_min) .* (xp ./ maximum(xp))
end

"""
    income_process(p) -> (z_vals, Π)

Rouwenhorst discretization of log productivity, exponentiated and normalised so
that the stationary mean of `z` is 1. Returns positive productivity levels and
the transition matrix.
"""
function income_process(p::SAGEParams)
    if p.z_vals_override !== nothing
        length(p.z_vals_override) == p.nz ||
            error("z_vals_override has length $(length(p.z_vals_override)) but nz = $(p.nz)")
        return copy(p.z_vals_override), copy(p.Π_override)
    end
    mc = rouwenhorst(p.nz, p.ρ, p.η)
    Π  = mc.p
    π_stat = stationary_distributions(mc)[1]
    z = exp.(mc.state_values)
    z ./= dot(π_stat, z)                # normalise E[z] = 1
    return collect(z), Matrix(Π)
end

# ----------------------------------------------------------------------------
# Solution container
# ----------------------------------------------------------------------------
struct SAGESolution
    p::SAGEParams
    a_grid::Vector{Float64}
    z_vals::Vector{Float64}
    Π::Matrix{Float64}
    # policies on the (na, nz) grid
    c::Matrix{Float64}          # consumption
    e::Matrix{Float64}          # effort
    q::Matrix{Float64}          # social contribution = 1 - e
    a_next::Matrix{Float64}     # next assets
    V::Matrix{Float64}          # value function
    λ::Matrix{Float64}          # stationary distribution over (a, z)
    # aggregates / wellbeing
    Q::Float64                  # public good size  A = Q = E[1-e]
    Uc::Matrix{Float64}         # material-gain utility at the optimum
    Us::Matrix{Float64}         # social-cohesion utility at the optimum
end

# material-gain (consumption) part of utility
@inline uc(c, e, p::SAGEParams) =
    p.Γ * (c^(1 - p.γ) / (1 - p.γ) - p.ϕ * e^(1 + p.ψ) / (1 + p.ψ))

# --- small numerical helpers (kept dependency-free) -------------------------
"Linear interpolation of `y` defined on sorted grid `x`, evaluated at `xq` (clamped)."
@inline function interp_lin(x::AbstractVector, y::AbstractVector, xq)
    n = length(x)
    xq <= x[1]  && return y[1]
    xq >= x[n]  && return y[n]
    k = searchsortedlast(x, xq)
    t = (xq - x[k]) / (x[k+1] - x[k])
    return (1 - t) * y[k] + t * y[k+1]
end

"Golden-section maximisation of unimodal `f` on [lo, hi]."
function golden_max(f, lo, hi; tol = 1e-9, maxit = 200)
    φ = (sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - φ * (b - a); d = a + φ * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:maxit
        if fc < fd
            a = c; c = d; fc = fd
            d = a + φ * (b - a); fd = f(d)
        else
            b = d; d = c; fd = fc
            c = b - φ * (b - a); fc = f(c)
        end
        (b - a) < tol && break
    end
    xstar = (a + b) / 2
    return xstar, f(xstar)
end

# Behavioural social-cohesion return to contributing q = (1-e). Zero in the
# canonical (:off) model, so effort is unaffected and policies are unchanged.
@inline function social_reward(e, Bz, p::SAGEParams, A_social)
    p.social_mode === :warmglow   ? p.social_strength * p.Λ * Bz * (1 - e) :
    p.social_mode === :multiplier ? p.social_strength * p.Λ * Bz * A_social * (1 - e) :
    0.0
end

# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
"""
    solve_model(p; method=PFI, A0=nothing) -> SAGESolution

Build the DiscreteDP, solve for policies, get the exact stationary distribution,
and compute the public good and the wellbeing dashboard.

Effort is collapsed into the reward: for each (state, next-asset) pair we pick
the effort on a grid that maximises within-period utility. The remaining choice
(next assets) is the single DiscreteDP action, exactly as in QuantEcon Aiyagari.

When `p.social_mode == :multiplier` the public good `A` raises the marginal
return to contributing, so the model is solved by an outer fixed point on `A`
(start it with `A0`). For `:off` and `:warmglow` there is no aggregate feedback
and the model is solved in one pass.
"""
function solve_model(p::SAGEParams; method = PFI, A0 = nothing,
                     damp = 0.5, tol = 1e-4, maxit = 80)
    p.social_mode === :multiplier || return _solve_once(p, 0.0; method = method)
    if p.homophily > 0
        # Per-group fixed point: the belonging aggregate of group g mixes the
        # economy-wide public good and the group's own mean contribution,
        #   A_eff[g] = (1-h)·Q + h·Qmean[g].
        h = p.homophily
        # A0 may be a scalar (symmetric start) or a per-group vector
        # (asymmetric start, to hunt for segregated equilibria)
        Qm = A0 === nothing ? fill(0.3, p.nz) :
             A0 isa Number  ? fill(Float64(A0), p.nz) : collect(Float64, A0)
        local sol
        converged = false
        for _ in 1:maxit
            sol = _solve_once(p, (1 - h) .* sol_total(Qm, p) .+ h .* Qm; method = method)
            Qm_new = group_means(sol)
            any(x -> isnan(x) || isinf(x), Qm_new) && break
            maximum(abs.(Qm_new .- Qm)) < tol && (Qm = Qm_new; converged = true; break)
            Qm = damp .* Qm .+ (1 - damp) .* Qm_new
        end
        converged || @warn "homophily fixed point did not converge to tol $tol in $maxit iterations"
        return sol
    end
    A = A0 === nothing ? 0.3 : A0           # outer fixed point on the public good
    local sol
    converged = false
    for _ in 1:maxit
        sol = _solve_once(p, A; method = method)
        (isnan(sol.Q) || isinf(sol.Q)) && break
        abs(sol.Q - A) < tol && (converged = true; break)
        A = damp * A + (1 - damp) * sol.Q
    end
    converged || @warn "multiplier fixed point did not converge to tol $tol in $maxit iterations (last |Q - A| = $(abs(sol.Q - A)))"
    return sol
end

"Group z-shares and mean contributions from a solution: Qmean[z] = E[1-e | z]."
group_means(s::SAGESolution) =
    [sum(s.λ[:, z] .* s.q[:, z]) / max(sum(s.λ[:, z]), 1e-300) for z in 1:s.p.nz]

"Total public good implied by group means under the stationary z-shares of `p`."
function sol_total(Qm::Vector{Float64}, p::SAGEParams)
    _, Π = income_process(p)              # respects any process override
    πz = stationary_distributions(MarkovChain(Π))[1]
    dot(πz, Qm)
end

"""
Single solve of the household problem given a fixed social aggregate.
`A_social` is a scalar (same belonging aggregate for everyone, the plain
multiplier) or a length-nz vector (per-group aggregate, the homophily case).
"""
function _solve_once(p::SAGEParams, A_social; method = PFI)
    Avec = A_social isa Number ? fill(Float64(A_social), p.nz) : A_social
    a = exponential_grid(p.a_min, p.a_max, p.na, p.pexp)
    z_vals, Π = income_process(p)
    na, nz = p.na, p.nz
    e_grid = range(0.0, 1.0, length = p.ne)

    n_s = na * nz                      # states s indexed (i_a, i_z) -> sidx
    sidx(i_a, i_z) = (i_z - 1) * na + i_a

    # Enumerate feasible (state, next-asset) pairs, folding the optimal effort
    # into the reward. Sparse state-action-pair DiscreteDP (QuantEcon idiom):
    # fast and memory-light, so the whole solve stays interactive-speed.
    Epol  = zeros(n_s, na)             # best effort for each (state, action)
    s_ind = Int[]; a_ind = Int[]; Rvec = Float64[]
    rows  = Int[]; cols = Int[]; vals = Float64[]   # sparse Q triplets
    pair  = 0
    for i_z in 1:nz
        z = z_vals[i_z]; α = p.α[i_z]
        for i_a in 1:na
            res_assets = p.R * a[i_a]
            s = sidx(i_a, i_z)
            for k in 1:na
                anext = a[k]
                best = -Inf; beste = 0.0
                Bz = p.B[i_z]
                @inbounds for e in e_grid
                    c = res_assets + (1 + p.subsidy) * α * e * z * p.Z - p.lumptax - anext
                    c <= 0 && continue
                    ut = uc(c, e, p) + social_reward(e, Bz, p, Avec[i_z])
                    if ut > best
                        best = ut; beste = e
                    end
                end
                best == -Inf && continue            # no feasible consumption
                Epol[s, k] = beste
                pair += 1
                push!(s_ind, s); push!(a_ind, k); push!(Rvec, best)
                for i_zn in 1:nz
                    push!(rows, pair); push!(cols, sidx(k, i_zn))
                    push!(vals, Π[i_z, i_zn])
                end
            end
        end
    end
    Qsp = sparse(rows, cols, vals, pair, n_s)

    ddp = DiscreteDP(Rvec, Qsp, p.β, s_ind, a_ind)
    results = solve(ddp, method)
    Vmat = reshape(results.v, na, nz)            # value function on the grid
    σ    = results.sigma                         # DiscreteDP next-asset action per state

    # --- Policy recovery --------------------------------------------------
    # EFFORT is taken from the DiscreteDP joint optimum over (a', e): it is the
    # consistent choice against the exact value function. Re-optimising effort
    # against the INTERPOLATED continuation value collapses it to a corner once
    # the social term is behavioural, so we do not do that. We then re-optimise
    # NEXT-ASSETS continuously with effort fixed, which keeps the savings policy
    # and the wealth distribution smooth (no grid sawtooth).
    EV = Vmat * Π'                               # EV[k, z] = E[V(a_k, z')|z]
    c = zeros(na, nz); e = zeros(na, nz); anext = zeros(na, nz)
    for i_z in 1:nz
        z = z_vals[i_z]; α = p.α[i_z]; Bz = p.B[i_z]
        evz = view(EV, :, i_z)
        for i_a in 1:na
            s  = sidx(i_a, i_z)
            ee = Epol[s, σ[s]]                   # effort from the DiscreteDP optimum
            cash0 = p.R * a[i_a]
            resources = cash0 + (1 + p.subsidy) * α * ee * z * p.Z - p.lumptax
            soc = social_reward(ee, Bz, p, Avec[i_z])
            hi = min(resources - 1e-10, a[end])
            if hi <= a[1]
                ba = a[1]
            else
                f = ap -> uc(resources - ap, ee, p) + soc + p.β * interp_lin(a, evz, ap)
                kbest = 1; vbest = -Inf
                @inbounds for k in 1:na
                    a[k] >= hi && break
                    v = f(a[k]); (v > vbest) && (vbest = v; kbest = k)
                end
                lo_b = a[max(kbest - 1, 1)]; hi_b = min(a[min(kbest + 1, na)], hi)
                ba = lo_b < hi_b ? golden_max(f, lo_b, hi_b)[1] : a[kbest]
            end
            anext[i_a, i_z] = ba
            e[i_a, i_z]     = ee
            c[i_a, i_z]     = resources - ba
        end
    end
    q = 1 .- e
    V = Vmat

    # --- Stationary distribution via Young (2010) lottery -----------------
    # Split each agent's continuous next-assets across the two bracketing grid
    # nodes for a smooth distribution. Next-assets are fixed, so build the
    # lottery transition ONCE as a sparse matrix and iterate with in-place
    # sparse multiplies (no per-iteration re-location or allocation).
    drows = Int[]; dcols = Int[]; dvals = Float64[]
    @inbounds for i_z in 1:nz, i_a in 1:na
        s = sidx(i_a, i_z)
        ap = anext[i_a, i_z]
        k = clamp(searchsortedlast(a, ap), 1, na - 1)
        w = clamp((a[k+1] - ap) / (a[k+1] - a[k]), 0.0, 1.0)
        for i_zn in 1:nz
            pz = Π[i_z, i_zn]
            push!(drows, s); push!(dcols, sidx(k,   i_zn)); push!(dvals, pz * w)
            push!(drows, s); push!(dcols, sidx(k+1, i_zn)); push!(dvals, pz * (1 - w))
        end
    end
    # Iterate the distribution to its stationary point with the prebuilt sparse
    # transition (fast per iteration, and robust to reducibility, where a direct
    # linear solve is numerically unstable). Slow-mixing wealth dynamics can need
    # many iterations, but each is a cheap sparse multiply.
    # Fast path: sparse power iteration (converges quickly for well-mixing
    # chains). Slow-mixing chains (spectral gap near zero, e.g. patient
    # households near the βR knife edge) stall here, so we fall back to dense
    # repeated squaring of the transition: T^(2^45) is ~3.5e13 effective
    # iterations, unconditionally stable for stochastic matrices (entries stay
    # in [0,1], rows renormalised against floating drift), and cheap because
    # the state space is only na*nz. This replaces both the old silent
    # non-convergence and the abandoned (unstable) direct linear solve.
    Tt = transpose(sparse(drows, dcols, dvals, n_s, n_s))   # λ' = Tᵀ λ
    λv = fill(1.0 / n_s, n_s); λn = similar(λv)
    dist_d = Inf
    for _ in 1:20_000
        mul!(λn, Tt, λv)
        d = 0.0
        @inbounds for i in eachindex(λv)
            d = max(d, abs(λn[i] - λv[i]))
        end
        λv, λn = λn, λv
        dist_d = d
        d < 1e-12 && break
    end
    if dist_d >= 1e-12
        Td = Matrix(transpose(Tt))                  # row-stochastic T, dense
        for _ in 1:45
            Td = Td * Td
            Td ./= sum(Td, dims = 2)                # keep rows stochastic
        end
        λv = vec(sum(Td .* (1.0 / n_s), dims = 1))  # uniform initial weights
    end
    λv ./= sum(λv)
    λ = reshape(λv, na, nz)

    Uc = zeros(na, nz); Us = zeros(na, nz)
    # public good A = Q = E[1 - e] over the stationary distribution
    Q = sum(λ .* q)

    # wellbeing dashboard at the optimum
    for i_z in 1:nz, i_a in 1:na
        Uc[i_a, i_z] = uc(c[i_a, i_z], e[i_a, i_z], p)
        Us[i_a, i_z] = p.Λ * p.B[i_z] * Q
    end

    return SAGESolution(p, a, z_vals, Π, c, e, q, anext, V, λ, Q, Uc, Us)
end

# ----------------------------------------------------------------------------
# Convenience diagnostics
# ----------------------------------------------------------------------------
"Fraction of agents at (essentially) the borrowing constraint (mass at lowest node)."
frac_constrained(s::SAGESolution; nodes = 1) =
    sum(s.λ[1:nodes, :])

"Share of the public good Q supplied by each income state (sums to 1)."
function public_good_shares(s::SAGESolution)
    contrib = [sum(s.λ[:, z] .* s.q[:, z]) for z in 1:s.p.nz]
    return contrib ./ sum(contrib)
end

"Gini of values `x` with population weights `w` (Lorenz-curve area)."
function _gini(x::AbstractVector, w::AbstractVector)
    ord = sortperm(x); x = x[ord]; w = w[ord]
    # prepend the Lorenz origin (0,0) so the first segment is included
    cw = vcat(0.0, cumsum(w)); cx = vcat(0.0, cumsum(w .* x)); cx ./= cx[end]
    1 - sum((cw[2:end] .- cw[1:end-1]) .* (cx[2:end] .+ cx[1:end-1]))
end

"Wealth (assets) Gini from the stationary distribution."
wealth_gini(s::SAGESolution) = _gini(vec(repeat(s.a_grid, 1, s.p.nz)), vec(s.λ))

"Income (labour income α·e·z) Gini from the stationary distribution. Wealth and
income inequality are distinct objects and can be reported and matched separately."
function income_gini(s::SAGESolution)
    inc = [s.p.α[iz] * s.e[ia, iz] * s.z_vals[iz] for ia in 1:s.p.na, iz in 1:s.p.nz]
    _gini(vec(inc), vec(s.λ))
end

end # module
