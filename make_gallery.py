#!/usr/bin/env python3
"""Build the notebook-gallery variant of the lecture.

The site source (index.md) uses MyST directives that GitHub and nbviewer do not
render: {cite} roles, {exercise}/{solution} gates, {contents}, {bibliography}.
This script rewrites them as plain Markdown, so the exported .ipynb reads
cleanly anywhere, then hands the result to jupytext.

  .venv/bin/python make_gallery.py          # writes gallery.md
  .venv/bin/jupytext --to ipynb gallery.md -o sage_bewley_wellbeing.ipynb
"""
import re
from pathlib import Path

SRC = Path(__file__).with_name("index.md")
OUT = Path(__file__).with_name("gallery.md")

TEXTUAL = {  # {cite:t}`key` -> Name (Year)
    "limademiranda2020": "Lima de Miranda and Snower (2020)",
    "havranek2015": "Havranek (2015)",
    "chetty2011": "Chetty, Guren, Manoli and Weber (2011)",
    "young2010": "Young (2010)",
    "kaplan2014": "Kaplan, Violante and Weidner (2014)",
    "conway2020": "Conway (2020)",
    "rouwenhorst1995": "Rouwenhorst (1995)",
}
PAREN = {  # {cite}`key` -> (Name Year)
    "conway2020": "(Conway 2020)",
    "bewley1986,aiyagari1994": "(Bewley 1986; Aiyagari 1994)",
    "rouwenhorst1995": "(Rouwenhorst 1995)",
    "insee2010": "(INSEE 2010)",
    "kaplan2014": "(Kaplan, Violante and Weidner 2014)",
    "limademiranda2020": "(Lima de Miranda and Snower 2020)",
}

REFERENCES = """Aiyagari, S. Rao (1994). Uninsured idiosyncratic risk and aggregate saving. *Quarterly Journal of Economics* 109(3), 659-684.

Bewley, Truman (1986). Stationary monetary equilibrium with a continuum of independently fluctuating consumers. In *Contributions to Mathematical Economics in Honor of Gerard Debreu*, North-Holland.

Chetty, Raj, Adam Guren, Day Manoli and Andrea Weber (2011). Are micro and macro labor supply elasticities consistent? *American Economic Review* 101(3), 471-475.

Conway, Alessandro (2020). Wellbeing and macroeconomics: a SAGE approach. Master's thesis, Sciences Po Paris.

Havranek, Tomas (2015). Measuring intertemporal substitution: the importance of method choices and selective reporting. *Journal of the European Economic Association* 13(6), 1180-1204.

INSEE (2010). Enquete Emploi du temps 2009-2010.

Kaplan, Greg, Giovanni L. Violante and Justin Weidner (2014). The wealthy hand-to-mouth. *Brookings Papers on Economic Activity*, Spring, 77-138.

Lima de Miranda, Katharina and Dennis J. Snower (2020). Recoupling economic and social prosperity. *Global Perspectives* 1(1).

Rouwenhorst, K. Geert (1995). Asset pricing implications of equilibrium business cycle models. In *Frontiers of Business Cycle Research*, Princeton University Press.

Young, Eric R. (2010). Solving the incomplete markets model with aggregate uncertainty using the Krusell-Smith algorithm and non-stochastic simulations. *Journal of Economic Dynamics and Control* 34(1), 36-41.
"""

text = SRC.read_text()

# {contents} block out
text = re.sub(r"```\{contents\}.*?```\n\n", "", text, flags=re.S)

# citation roles to plain text
for key, rep in TEXTUAL.items():
    text = text.replace("{cite:t}`%s`" % key, rep)
for key, rep in PAREN.items():
    text = text.replace("{cite}`%s`" % key, rep)

# exercises: numbered bold headers
ex_n = iter(range(1, 10))
text = re.sub(r"```\{exercise\}\n:label: [^\n]+\n\n(.*?)```",
              lambda m: "**Exercise %d.** %s" % (next(ex_n), m.group(1).rstrip()) + "\n",
              text, flags=re.S)
text = re.sub(r"```\{solution-start\}[^\n]*\n:class: dropdown\n```", "**Solution.**", text)
text = re.sub(r"```\{solution-end\}\n```\n?", "", text)

# bibliography directive to the plain list
text = re.sub(r"```\{bibliography\}\n```", REFERENCES.strip(), text)

assert "{cite" not in text, "unconverted citation left"
assert "{exercise" not in text and "{solution" not in text, "unconverted directive left"

OUT.write_text(text)
print(f"wrote {OUT.name} ({len(text.splitlines())} lines)")
