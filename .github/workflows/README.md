# GitHub Actions Workflows

This directory contains the CI/CD pipelines for Neuro-INT Mamba.

## Workflows

### 1. CI (`ci.yml`)
- **Trigger**: Every push to `main` and every Pull Request.
- **Purpose**: Ensures code quality and functional correctness.
- **Steps**:
    - **Linting**: Uses `ruff` to check code style.
    - **Type Checking**: Uses `ty` for static type analysis.
    - **Testing**: Runs the `pytest` suite.

### 2. Publish to PyPI (`publish.yml`)
- **Trigger**: When a new tag starting with `v` (e.g., `v0.1.0`) is pushed.
- **Purpose**: Automates the release process to PyPI.
- **Steps**:
    - Builds the source distribution and wheel using `uv build`.
    - Uploads to PyPI using `uv publish`.
- **Requirements**: Requires a GitHub Secret named `PYPI_TOKEN` containing a valid PyPI API token.

## How to Release a New Version

1. Update the version in `pyproject.toml`.
2. Commit and push the change.
3. Create and push a new tag:
   ```bash
   git tag v0.1.x
   git push origin v0.1.x
   ```
4. Monitor the "Actions" tab for the "Publish to PyPI" workflow.
