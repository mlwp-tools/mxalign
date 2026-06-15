"""Built-in metric implementations with a fused-engine fast path.

Each metric here is a regular Python function that operates on xarray
objects — usable under ``engine: xarray`` exactly like ``scores.continuous.*``
or ``xskillscore.*``. They are additionally decorated with
:func:`mxalign.fused_metric`, attaching a numpy kernel + finalizer that the
fused verification engine discovers and uses on the per-reference_time
fast path.

YAML usage::

    metrics:
      mse:
        function: mxalign.scores.mse
        inputs: {fcst: forecast, obs: reference}
        reduce_dims: [reference_time]

Switching to a different backend (``scores.continuous.mse``,
``xskillscore.mse``, ...) stays a one-line config change; the fused fast
path is only available when ``function:`` resolves to a function decorated
with ``@fused_metric``.
"""
from __future__ import annotations

import numpy as np

from .verification import fused_metric


def _np32(a):
    return np.asarray(a, dtype=np.float32) if not isinstance(a, np.ndarray) else (
        a if a.dtype == np.float32 else a.astype(np.float32, copy=False)
    )


def _kernel_squared_error(fcst, obs):
    diff = _np32(fcst) - _np32(obs)
    return diff * diff


def _kernel_absolute_error(fcst, obs):
    return np.abs(_np32(fcst) - _np32(obs))


def _kernel_error(fcst, obs):
    return _np32(fcst) - _np32(obs)


def _finalize_mean(partial_sum, n):
    return partial_sum / n


@fused_metric(kernel=_kernel_squared_error, finalize=_finalize_mean)
def mse(fcst, obs, reduce_dims=None, **_):
    """Mean squared error along ``reduce_dims``."""
    diff = fcst - obs
    return (diff * diff).mean(dim=reduce_dims)


@fused_metric(kernel=_kernel_absolute_error, finalize=_finalize_mean)
def mae(fcst, obs, reduce_dims=None, **_):
    """Mean absolute error along ``reduce_dims``."""
    return abs(fcst - obs).mean(dim=reduce_dims)


@fused_metric(kernel=_kernel_error, finalize=_finalize_mean)
def bias(fcst, obs, reduce_dims=None, **_):
    """Mean error (forecast minus observation) along ``reduce_dims``."""
    return (fcst - obs).mean(dim=reduce_dims)


# Alias: ``mean_error`` is the same thing as ``bias`` in this context, kept
# so existing YAMLs that say ``function: scores.continuous.mean_error`` can
# migrate to ``function: mxalign.scores.mean_error`` without semantic drift.
mean_error = bias


# ---------------------------------------------------------------------------
# NVV group-pattern helpers (also imported by the fused_collect engine)
# ---------------------------------------------------------------------------

def _extract_captures(pattern, ds_vars):
    """Return the set of ``*`` captures for a single-wildcard pattern.

    Given a pattern such as ``"*_x"`` and a collection of variable names,
    returns every string ``c`` such that ``pattern.replace("*", c)`` is
    present in ``ds_vars``.  Literal patterns (no ``*``) return
    ``{pattern}`` if the name exists, else an empty set.
    """
    if "*" not in pattern:
        return {pattern} if pattern in ds_vars else set()
    idx = pattern.index("*")
    prefix, suffix = pattern[:idx], pattern[idx + 1:]
    if "*" in suffix:
        raise ValueError(
            f"Multiple wildcards are not supported in component patterns: {pattern!r}"
        )
    captures: set[str] = set()
    for v in ds_vars:
        if not v.startswith(prefix):
            continue
        rest = v[len(prefix):]
        if suffix:
            if not rest.endswith(suffix):
                continue
            captures.add(rest[: -len(suffix)])
        else:
            captures.add(rest)
    return captures


def _expand_group_patterns(variables_cfg, ds_vars):
    """Expand glob-keyed entries in an NVV ``variables:`` config dict.

    A group is treated as a glob entry when its key or any component
    pattern contains ``*``.  For each such entry, all values of the
    wildcard are found such that **every** component pattern evaluates to
    an existing variable name (present in ``ds_vars``); a concrete group
    is created for each valid capture.

    Literal-keyed entries (no ``*`` in key or components) are kept as-is
    and override any glob-generated group that has the same resolved name.

    Parameters
    ----------
    variables_cfg : dict
        Raw ``variables:`` dict from the YAML metric config.
    ds_vars : iterable of str
        Variable names available in the dataset at evaluation time.

    Returns
    -------
    dict
        Fully-resolved ``{group_name: {"components": [...], ...}}`` dict
        suitable for direct use in :func:`nvv`.
    """
    ds_set = set(ds_vars)
    literal: dict = {}
    globs: list = []
    for key, grp_cfg in variables_cfg.items():
        comps = grp_cfg.get("components", [])
        if "*" in key or any("*" in c for c in comps):
            globs.append((key, grp_cfg))
        else:
            literal[key] = grp_cfg

    expanded: dict = {}
    for key_tmpl, grp_cfg in globs:
        comps = grp_cfg.get("components", [])
        # Intersection of captures that satisfy ALL component patterns.
        valid: set[str] | None = None
        for comp in comps:
            caps = _extract_captures(comp, ds_set)
            valid = caps if valid is None else valid & caps
        if not valid:
            continue
        extra = {k: v for k, v in grp_cfg.items() if k != "components"}
        for cap in sorted(valid):
            name = key_tmpl.replace("*", cap)
            expanded[name] = {"components": [c.replace("*", cap) for c in comps], **extra}

    expanded.update(literal)  # literal entries override glob-generated ones
    return expanded


def nvv(fcst, obs, reduce_dims=None, components=None, variables=None,
        eps=0.0, label=None, **_):
    """Normalized Vector Variance.

    Compares the temporal spread of one or more vector fields in forecast vs
    observation.  For each group of components ``i``, the vector variance is::

        VV = sqrt( sum_i Var(x_i) )

    where ``Var(x_i)`` is the variance of component ``i`` along
    ``reduce_dims``.  The NVV is::

        NVV = VV_fcst / max(VV_obs, eps)

    Two calling forms are supported:

    **Single group** (``components`` key)::

        metrics:
          nvv_wind10m:
            function: mxalign.scores.nvv
            inputs: {fcst: forecast, obs: reference}
            reduce_dims: [reference_time]
            components: [10u, 10v]
            label: wind10m        # optional; default "10u+10v"

    **Multiple groups** (``variables`` key)::

        metrics:
          nvv:
            function: mxalign.scores.nvv
            inputs: {fcst: forecast, obs: reference}
            reduce_dims: [reference_time, grid_index]
            variables:
              z_500_grad:
                components: [z_500_grad_x, z_500_grad_y]
              q_500_grad:
                components: [q_500_grad_x, q_500_grad_y]
                eps: 1.0e-5     # per-group override (optional)

    Parameters
    ----------
    fcst, obs : xr.Dataset
        Forecast and observation datasets.
    reduce_dims : str or list of str
        Dimension(s) to reduce over (e.g. ``["reference_time"]``).
    components : list of str
        Single-group form: variable names forming the vector.
    variables : dict
        Multi-group form: ``{label: {components: [...], eps: ...}, ...}``.
        Top-level ``eps`` is the default for any group that omits it.
    eps : float
        Zero-guard threshold, in the units of VV (not VV squared). NVV is set
        to NaN wherever ``VV_obs <= eps``. Default ``0.0`` masks only an
        exactly-zero observation spread. Note: this is *not* a floor added to
        the denominator (which would corrupt small-magnitude fields such as
        specific-humidity gradients); it only masks undefined ratios.
    label : str, optional
        Coordinate value for the synthetic ``variable`` dim in single-group
        form. Defaults to ``"+".join(components)``.

    Returns
    -------
    xr.DataArray
        Shape ``(variable, ...)`` where ``variable`` has one entry per group
        and ``...`` are whatever dims remain after reducing over
        ``reduce_dims``.
    """
    import xarray as xr

    # ---- normalise to the grouped form ------------------------------------
    if variables is not None:
        ds_vars = list(fcst.data_vars) if isinstance(fcst, xr.Dataset) else []
        # Expand glob patterns (e.g. "*": {components: ["*_x", "*_y"]}).
        needs_expansion = any(
            "*" in k or any("*" in c for c in v.get("components", []))
            for k, v in variables.items()
        )
        resolved = _expand_group_patterns(variables, ds_vars) if needs_expansion else variables
        groups = {
            grp_label: {
                "components": grp_cfg["components"],
                "eps": grp_cfg.get("eps", eps),
            }
            for grp_label, grp_cfg in resolved.items()
        }
    elif components:
        coord = label if label is not None else "+".join(components)
        groups = {coord: {"components": components, "eps": eps}}
    else:
        raise ValueError(
            "mxalign.scores.nvv requires either `components:` (single group) "
            "or `variables:` (multiple groups) to be set."
        )

    rd = [reduce_dims] if isinstance(reduce_dims, str) else list(reduce_dims or [])

    results = {}
    for grp_label, grp_cfg in groups.items():
        comps = grp_cfg["components"]
        grp_eps = grp_cfg["eps"]

        vv_sq_fcst = None
        vv_sq_obs = None
        for var in comps:
            s2_fcst = (fcst[var] if isinstance(fcst, xr.Dataset) else fcst).var(dim=rd)
            s2_obs = (obs[var] if isinstance(obs, xr.Dataset) else obs).var(dim=rd)
            vv_sq_fcst = s2_fcst if vv_sq_fcst is None else vv_sq_fcst + s2_fcst
            vv_sq_obs = s2_obs if vv_sq_obs is None else vv_sq_obs + s2_obs

        vv_fcst = vv_sq_fcst ** 0.5
        vv_obs = vv_sq_obs ** 0.5
        results[grp_label] = (vv_fcst / vv_obs).where(vv_obs > grp_eps)

    return xr.Dataset(results)
