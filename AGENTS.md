# AGENTS.md

Guidance for AI coding agents and human contributors working on `mxalign`.

## Project

`mxalign` aligns meteorological `xarray` datasets in space and time. Datasets carry
**traits** (`mlwp_space_trait`, `mlwp_time_trait`, `mlwp_uncertainty_trait` in `ds.attrs`).
The traits are defined by [mlwp-data-specs](https://github.com/mlwp-tools/mlwp-data-specs) and
attached by [mlwp-data-loaders](https://github.com/mlwp-tools/mlwp-data-loaders). Loading and
validating datasets happens in those packages. `mxalign` only reads the traits to decide how to
align.

Package layout (`src/mxalign/`):

| Package | Contents |
|---|---|
| `accessor/` | The `ds.mx` accessor (`MxAccessor`): trait predicates (`is_grid`, `is_forecast`, ...) and thin wrappers around alignment operations |
| `alignment/` | Time and space alignment (pairwise ops and the multi-dataset `align_time` / `align_space`), NaN broadcasting, and `interpolations/` (interpolator registry, xarray and Delaunay interpolators) |
| `pipeline/` | The YAML-driven pipeline: `cli` (the `mxalign` command), `runner`, `verification`, `transformations/`, and `config` (pydantic schema for the YAML config) with its `dates` / `save` helpers |
| `utils/` | Helpers shared by `accessor` and `alignment` (`projections`, `traits`) |

Public functions are re-exported from `mxalign/__init__.py`. Keep those re-exports working when
moving code.

## Development tools

- Python >= 3.12, managed with [uv](https://docs.astral.sh/uv/).
- Set up the environment: `uv sync --all-extras`
- Run the tests: `uv run pytest tests -v`
- Run the linters: `uvx pre-commit run --all-files`. This runs ruff check (with `--fix`), ruff
  format, codespell, numpydoc-validation and the standard pre-commit hooks. CI runs the same hooks
  (`.github/workflows/linting.yaml`), plus the tests (`.github/workflows/tests.yaml`).
- Add a dependency with `uv add <package>` (or `uv add --optional <extra> <package>`), and commit
  both `pyproject.toml` and `uv.lock`.

## Coding conventions

- **Type annotations** on every argument (including `*args` / `**kwargs`) and every return value,
  in `src/` and in `tests/`. Use modern syntax (`X | None`, `list[str]`) and concrete types
  (`xr.Dataset`, `np.ndarray`, ...) rather than `Any`. Enforced by ruff's `ANN` rules.
- **Numpy-style docstrings** on every module, class, function and method, in `src/` and in
  `tests/`. Each one has a one-line imperative summary ending in a period, then `Parameters`,
  `Returns` (or `Yields`), and `Raises` where applicable. Class docstrings document the
  `__init__` parameters, so `__init__` itself needs no docstring (it is excluded from
  numpydoc's checks). pydantic models in `pipeline/config.py` document their fields under
  `Attributes`. Enforced by ruff's `D` rules (numpy convention) and by `numpydoc lint`.
  `Examples`, `See Also` and an extended summary are optional. Run `ruff` through pre-commit, which
  pins the version: newer ruff releases enable more rules by default.

  ```python
  def align_grid_point(
      ds1: xr.Dataset, ds2: xr.Dataset, method: str = "xarray", **kwargs: object
  ) -> xr.Dataset:
      """Interpolate grid dataset ``ds1`` to the point locations of ``ds2``.

      Parameters
      ----------
      ds1 : xr.Dataset
          Gridded source dataset.
      ds2 : xr.Dataset
          Point dataset whose locations are the interpolation targets.
      method : str, default "xarray"
          Name of a registered interpolator.
      **kwargs : object
          Passed to the interpolator's constructor.

      Returns
      -------
      xr.Dataset
          ``ds1`` interpolated to the points of ``ds2``.
      """
  ```
- **Formatting** with `ruff format`. Don't hand-format against it.
- **Never mutate inputs**: alignment operations and accessor methods return new datasets (see
  `MxAccessor`, which reads traits from `ds.attrs` on every call and never modifies `ds`).
- **Registries**: new interpolators and transformations are registered with
  `register_interpolator` / `register_transformation` and imported in their package
  `__init__.py`, so they register on `import mxalign`.
- **Config**: new pipeline config keys go into the pydantic models in
  `pipeline/config.py`. Sections reject unknown keys unless they pass kwargs through to a function.

## Tests

- pytest, under `tests/`. Shared fixtures (`ds_fcst`, `ds_obs`, ...) are in `tests/conftest.py`.
- Test functions are annotated (`-> None`, fixture arguments typed) and have numpy-style docstrings
  that state the behaviour being checked, the same as package code.
- Add or update tests together with any behaviour change.

## Changes and decisions

- Add an entry to `CHANGELOG.md` under `[Unreleased]` (Keep a Changelog format) with your GitHub
  handle. Mark breaking changes with **Breaking:**.
- Record design decisions as ADRs in `doc/adr/` using `doc/adr/template.md`.
