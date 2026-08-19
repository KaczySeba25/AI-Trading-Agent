# CI

`github-actions-ci.yml` is a ready-to-use GitHub Actions workflow. It is parked
here rather than in `.github/workflows/` because pushes that create or modify
workflow files are rejected unless the pushing identity holds the `workflows`
permission:

```
refusing to allow a GitHub App to create or update workflow
`.github/workflows/ci.yml` without `workflows` permission
```

## Enabling it

```bash
mkdir -p .github/workflows
cp ci/github-actions-ci.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "ci: enable GitHub Actions"
git push
```

Push that from your own account (not through an app integration) and it will
run on the next push and pull request.

## What it checks

| Step | Purpose |
| --- | --- |
| `check_no_secrets.py` | Fails the build if a high-entropy credential is committed. |
| `ruff check` | Style and obvious bugs. Advisory (`continue-on-error`) so lint drift never blocks a fix. |
| `pytest -q` | The full suite: 56 tests, no network, no torch. |
| CLI smoke test | Generates data and runs `backtest` + `paper` so import and wiring errors surface. |

The workflow deliberately installs **without** `torch` and
`stable-baselines3`. `agent_system/rl/agent.py` falls back to the heuristic
policy when they are missing, and CI is the best place to keep that fallback
honest. To test the PPO path too, add `pip install torch stable-baselines3
gymnasium` to the install step and expect the job to take several minutes
longer.

Nothing in CI touches the network or Binance: every mode used here is offline
and runs on the synthetic CSV.
