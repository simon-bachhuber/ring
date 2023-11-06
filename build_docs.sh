# execute in root dir of project with `sh build_docs.sh`

rm docs/index.md
cp readme.md docs/index.md
mkdocs gh-deploy