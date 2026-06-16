# Contributing to Quint

Thanks for your interest in improving Quint! 🎙️

## Getting started

```shell
git clone https://github.com/poloniki/quint.git
cd quint
make install        # pip install -e .
```

Copy the environment template and add your keys:

```shell
cp env.sample .env  # set OPENAI_API_KEY, etc.
```

## Development workflow

```shell
make run_api        # serve the API locally on :8083 with reload
make test           # run the test suite (pytest)
make lint           # static checks (ruff)
make format         # auto-format (ruff)
```

We use [pre-commit](https://pre-commit.com/) to keep formatting and linting
consistent:

```shell
pip install pre-commit
pre-commit install
```

## Opening a pull request

1. Branch off `master`.
2. Keep changes focused; add or update tests where it makes sense.
3. Make sure `make lint` and `make test` pass.
4. Open a PR with a clear description of the change.

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE).

## Releasing to PyPI

The package is published to PyPI as [`quintessentia`](https://pypi.org/project/quintessentia/)
(the import name stays `quint`). Releases use **Trusted Publishing** (OIDC) — no
API tokens are stored. `.github/workflows/publish.yml` builds and uploads the
distribution whenever a GitHub Release is published.

**One-time setup** (maintainers only):

1. On PyPI → [Publishing](https://pypi.org/manage/account/publishing/), add a
   *pending publisher* with: project `quintessentia`, owner `poloniki`,
   repository `quint`, workflow `publish.yml`, environment `pypi`.
2. On GitHub → Settings → Environments, create an environment named `pypi`.

**Cutting a release:**

1. Bump `version=` in `setup.py`.
2. Optionally check the build locally: `pip install build twine && python -m build && twine check dist/*`.
3. Create a GitHub Release with a tag (e.g. `v1.2`). The publish workflow runs
   automatically and uploads to PyPI.
