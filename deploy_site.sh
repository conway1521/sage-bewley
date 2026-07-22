#!/usr/bin/env bash
# Deploy the built lecture to this repository's gh-pages branch (GitHub Pages).
# Run AFTER building the book (see DEPLOY.md). Publishes _build/html as the
# site root, with .nojekyll so the underscore-prefixed asset folders are served.
#
#   bash deploy_site.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SITE="$HERE/_build/html"
REMOTE="https://github.com/conway1521/sage-bewley.git"

if [ ! -f "$SITE/index.html" ]; then
  echo "No built site at $SITE. Build first (see DEPLOY.md)."; exit 1
fi

TMP="$(mktemp -d)"
git clone --quiet "$REMOTE" "$TMP"
cd "$TMP"
git checkout --quiet --orphan gh-pages
git rm -rf --quiet . >/dev/null 2>&1 || true
cp -R "$SITE/." .
touch .nojekyll
git add -A
git -c user.name="Alessandro Conway" -c user.email="colt.00.caving@icloud.com" \
    commit --quiet -m "Deploy lecture site $(date +%Y-%m-%d)"
git push --quiet -f origin gh-pages
echo "Deployed. Live at https://conway1521.github.io/sage-bewley/ once Pages is enabled."
