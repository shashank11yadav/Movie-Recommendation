# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/shashank11yadav/Movie-Recommendation/security/advisories/new)
rather than a public issue. I will acknowledge a report within a week.

## Scope

The deployed site is a static bundle (HTML, JS, CSS, JSON) on GitHub Pages. It has
no server, no accounts, no cookies, no API keys and no third-party scripts, so
the realistic risks are in the build and supply chain:

- Dependencies are pinned (`web/package-lock.json`, `pipeline/uv.lock`) and
  updated by Dependabot. CI runs `npm audit --omit=dev` and `pip-audit`.
- GitHub Actions are pinned to commit SHAs and run with `contents: read` unless
  a job needs more (Pages deploy, CodeQL upload).
- The pipeline downloads MovieLens over HTTPS, verifies a pinned SHA-256 and
  extracts only known file names. The embedding model is pinned to a Hugging
  Face commit and loaded from safetensors with remote code disabled.
- The frontend validates every JSON file and the URL fragment before use and
  ships a Content-Security-Policy that allows only same-origin scripts, styles
  and fetches.
