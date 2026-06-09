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
