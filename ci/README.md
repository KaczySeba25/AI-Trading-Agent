# Continuous integration

`github-actions-ci.yml` is a ready-to-use workflow: it installs the dependencies,
scans for committed secrets, byte-compiles the package and runs the test suite on
Python 3.11 and 3.12.

It lives here rather than in `.github/workflows/` because the GitHub App used to push
this branch is not granted the `workflows` permission, so it cannot create workflow
files. Activate it yourself with:

```bash
mkdir -p .github/workflows
cp ci/github-actions-ci.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml && git commit -m "Enable CI" && git push
```
