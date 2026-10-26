# Support

Synthetic-data comparisons of logistic regression, XGBoost, MLP and GCN on stock-correlation and simulated fund-holding graphs.

## Start here

Read the [README](README.md) for setup, examples and method boundaries. The tested maintenance commands are defined in the [CI workflow](.github/workflows/ci.yml). Environment: Python 3.10+; CPU PyTorch is sufficient for the smoke configuration.

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
```

Start with the CPU smoke configuration before trying the default 300-stock experiment. Include memory and device details for training failures. The published three-seed experiment has not shown a stable GCN ranking advantage; do not rewrite this as a positive real-market result. Keep private holdings and licensed datasets out of public reports.

## Ask a question or report a problem

Use [GitHub Issues](https://github.com/dev-belly/investor-network-gnn/issues) for usage questions, reproducible bugs and scoped feature proposals. English and Chinese reports are welcome. Check existing issues first and use the bug or feature form when it fits. Support is best effort; no response time or delivery date is promised.

Include the config, seed, label horizon, purge gap, split dates, graph construction mode, backend and saved prediction/checkpoint identifiers. Preserve all preselected seeds and compare the same checkpoint when reporting test-edge removal.

Include the commit SHA, operating system, dependency versions, exact command, expected behavior and a minimal reproduction. Share only redacted logs and data you are allowed to publish. Remove credentials, personal information and confidential or licensed records. For a suspected security vulnerability, follow [SECURITY.md](SECURITY.md) before posting details.

## Verify locally

Run relevant checks before reporting a regression or opening a pull request:

```bash
python -m pytest tests/ -q
python scripts/run_pipeline.py --config configs/smoke.yaml --force-data --run-dir .ci-smoke
python experiments/2026-09-28/summarize.py
```

These are verification commands, not a claim that a new environment has already passed them. For changes that affect results, retain input provenance, configuration and seeds alongside the output.
