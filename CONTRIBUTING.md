# Contributing to pde-relaxation

Thanks for contributing!

Local checks
- Create a virtualenv & install deps:
  python -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
- Format: python -m black .
- Run tests: pytest -q

Pre-commit
- Install hooks locally:
  pip install pre-commit
  pre-commit install
  # Run hooks once on all files
  pre-commit run --all-files

PR process
1. Create a feature branch from feature/integrations (or feature/enhancements).
2. Open a PR with a clear title and description; link related issues.
3. Fix CI issues and address review comments; merge after approvals and green CI.

Notes
- Mark heavy or GPU tests with @pytest.mark.slow or @pytest.mark.gpu and avoid running them in CI unless configured.
- Prefer pinning external git dependencies to commit SHAs for reproducible CI.
