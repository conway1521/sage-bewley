# Deploying the lecture

The lecture is a Jupyter Book. It is published to GitHub Pages from the
`gh-pages` branch of this repository, so it goes live at

  https://conway1521.github.io/sage-bewley/

## One-time setup (you, in the browser, once)

Enable Pages on the repository:

1. Go to https://github.com/conway1521/sage-bewley/settings/pages
2. Under "Build and deployment", set Source to "Deploy from a branch".
3. Set Branch to `gh-pages` and folder to `/ (root)`. Save.
4. Wait a minute, then load https://conway1521.github.io/sage-bewley/

The `gh-pages` branch is already pushed, so the site appears as soon as Pages
is enabled. The `.nojekyll` file is in place, which is what makes the
underscore-prefixed asset folders (`_static`, `_images`) load.

## Updating the lecture later

1. Edit `index.md` (or `sage_engine.jl`).
2. Rebuild the book (the Julia kernel must be registered; see README):

   ```
   .venv/bin/jupyter-book build .
   ```

3. Redeploy:

   ```
   bash deploy_site.sh
   ```

`deploy_site.sh` clones the repo to a temporary directory, replaces the
`gh-pages` branch with the freshly built `_build/html`, and force-pushes. Your
working tree is never touched.

