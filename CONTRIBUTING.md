# Contributing to Investor Network GCN

Synthetic-data comparisons of logistic regression, XGBoost, MLP and GCN on stock-correlation and simulated fund-holding graphs.

Bug reports, documentation fixes and focused improvements are welcome. English and Chinese are both fine. For setup and questions, start with [SUPPORT.md](SUPPORT.md) and the [README](README.md). Follow [SECURITY.md](SECURITY.md) for suspected vulnerabilities.

## Prepare a change

Check existing issues and pull requests to avoid duplicating work. For a larger design or research change, describe the problem, assumptions and acceptance criteria in an issue first. Keep each pull request focused and explain the resulting behavior for a reader who has not followed the discussion.

Use Python 3.10+; CPU PyTorch is sufficient for the smoke configuration.

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
```

Activate the virtual environment with the command appropriate for your shell. Run commands from the repository root unless a command changes directory.

## Validate

The [CI workflow](.github/workflows/ci.yml) is the current reference for automated checks. Run the checks relevant to your change and report the actual outcome:

```bash
python -m pytest tests/ -q
python scripts/run_pipeline.py --config configs/smoke.yaml --force-data --run-dir .ci-smoke
python experiments/2026-09-28/summarize.py
```

Add or update a meaningful regression test when behavior changes. A documentation-only change needs review of its commands, links and claims; do not invent test outcomes or benchmark results. If a check was skipped, say so in the pull request.

## Preserve evidence

Include the config, seed, label horizon, purge gap, split dates, graph construction mode, backend and saved prediction/checkpoint identifiers. Preserve all preselected seeds and compare the same checkpoint when reporting test-edge removal.

Start with the CPU smoke configuration before trying the default 300-stock experiment. Include memory and device details for training failures. The published three-seed experiment has not shown a stable GCN ranking advantage; do not rewrite this as a positive real-market result. Keep private holdings and licensed datasets out of public reports.

Keep historical results distinguishable from reruns. Record configurations, seeds, environment and source revision when adding experiments. Avoid choosing favorable seeds or changing benchmark definitions to improve a headline number. Describe limitations and label proposed features as planned until implemented and verified.

## Submit

Use a descriptive branch and commit message, fill in the pull request template, and link any related issue. Include concise validation evidence and update relevant documentation. Do not include generated local caches, credentials, private datasets or unrelated edits. Contributions are reviewed as capacity permits; no response or merge date is promised.
