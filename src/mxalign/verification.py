from .transformations.external import _resolve_function
from functools import partial

import numpy as np


# ---------------------------------------------------------------------------
# Sum-decomposable kernels & finalizers (opt-in, config-driven)
# ---------------------------------------------------------------------------
# A metric is "sum-decomposable along reference_time" if its full result can
# be obtained by:
#   1. computing a per-sample partial array via ``kernel(fcst, ref)``,
#   2. summing partials across reference_time,
#   3. applying ``finalize(partial_sum, n_samples)`` once at the end.
#
# Kernels and finalizers are independent registries keyed by short
# mathematical names. Whether (and how) a metric uses them is decided in the
# YAML per metric, e.g.:
#
#   metrics:
#     mse:
#       function: scores.continuous.mse    # backend choice — config-level
#       kernel: squared_error              # opt-in to the fused fast path
#       finalize: mean                     # default if omitted
#
# Switching backends (e.g. ``xskillscore.mse`` ↔ ``scores.continuous.mse``)
# does not require touching the registry: the kernel is determined by the
# math, not the implementation.
#
# Both registries are public; downstream code may register custom kernels or
# finalizers (e.g. ``register_finalize("rms", lambda s, n: np.sqrt(s/n))``).

_KERNEL_REGISTRY: dict[str, "callable"] = {}
_FINALIZE_REGISTRY: dict[str, "callable"] = {}


def register_kernel(name, fn):
    """Register a per-sample kernel ``fn(fcst, ref) -> np.ndarray`` under
    ``name``. The output must be summable along the reduction dimension."""
    _KERNEL_REGISTRY[name] = fn


def register_finalize(name, fn):
    """Register a finalizer ``fn(partial_sum, n_samples) -> result`` under
    ``name``."""
    _FINALIZE_REGISTRY[name] = fn


def get_kernel(name):
    """Return the kernel callable for ``name``, or ``None`` if unknown."""
    return _KERNEL_REGISTRY.get(name)


def get_finalize(name):
    """Return the finalizer callable for ``name``, or ``None`` if unknown."""
    return _FINALIZE_REGISTRY.get(name)


def _kernel_squared_error(fcst, ref):
    diff = fcst.astype(np.float32, copy=False) - ref.astype(np.float32, copy=False)
    return diff * diff


def _kernel_absolute_error(fcst, ref):
    return np.abs(
        fcst.astype(np.float32, copy=False) - ref.astype(np.float32, copy=False)
    )


def _kernel_error(fcst, ref):
    return fcst.astype(np.float32, copy=False) - ref.astype(np.float32, copy=False)


register_kernel("squared_error", _kernel_squared_error)
register_kernel("absolute_error", _kernel_absolute_error)
register_kernel("error", _kernel_error)

register_finalize("mean", lambda partial_sum, n: partial_sum / n)
register_finalize("sum", lambda partial_sum, n: partial_sum)


# ---------------------------------------------------------------------------
# Fused fast-path marker
# ---------------------------------------------------------------------------
# A metric function may declare a sum-decomposable fast path by being
# decorated with ``@fused_metric(kernel=..., finalize=...)``. The decorator
# attaches two attributes to the function object:
#
#   * ``_fused_kernel(fcst, ref) -> np.ndarray``  — per-sample partial,
#     summable along the reduction dimension.
#   * ``_fused_finalize(partial_sum, n_samples) -> result`` — final
#     reduction (e.g. divide by ``n`` for means).
#
# The fused verification engine discovers the fast path by inspecting these
# attributes on the function resolved from the YAML's ``function:`` field;
# no registry is involved. The decorated function itself must remain a
# valid xarray-side metric so it works under ``engine: xarray`` too.

def fused_metric(*, kernel, finalize):
    """Mark a metric function as having a fused-engine fast path.

    Parameters
    ----------
    kernel : callable
        ``kernel(fcst, ref) -> np.ndarray`` returning a per-sample partial
        that is summable across the reduction dimension. Operates on plain
        numpy arrays of identical shape.
    finalize : callable
        ``finalize(partial_sum, n_samples) -> result`` reducing the
        accumulated sum (e.g. ``lambda s, n: s / n`` for means).
    """

    def decorator(fn):
        fn._fused_kernel = kernel
        fn._fused_finalize = finalize
        return fn

    return decorator


def get_fused_kernel(fn):
    """Return ``(kernel, finalize)`` for a function decorated with
    :func:`fused_metric`, or ``None`` if the function has no fast path.
    """
    kernel = getattr(fn, "_fused_kernel", None)
    finalize = getattr(fn, "_fused_finalize", None)
    if kernel is None or finalize is None:
        return None
    return kernel, finalize


class Metric:
    def __init__(self, name, func_path, ds_ref, inputs, **kwargs):
        self.name = name
        self.func_path = func_path
        # Opt-in sum-decomposable fields. Popped from kwargs so they are
        # never forwarded to the metric function itself. Validation of the
        # registered names is deferred until ``.kernel`` / ``.finalize`` is
        # actually read, so the legacy path (which doesn't care) keeps
        # working regardless of typos.
        self._kernel_name = kwargs.pop("kernel", None)
        self._finalize_name = kwargs.pop("finalize", "mean")

        func = _resolve_function(func_path)
        self._is_xskillscore = func.__module__.startswith("xskillscore")
        self._dim = kwargs.get("dim", None)

        kwarg_ref = {}
        kwarg_ds = []
        for input_arg, ds_type in inputs.items():
            if ds_type == "reference":
                kwarg_ref[input_arg] = (
                    self._rechunk(ds_ref) if self._is_xskillscore else ds_ref
                )
            else:
                kwarg_ds.append(input_arg)
        if len(kwarg_ds) > 1:
            raise ValueError(
                f"More than one predictor-input argument defined for function {func_path}"
            )
        partial_kwargs = {**kwarg_ref, **kwargs}
        self._func = partial(func, **partial_kwargs)
        self._kwarg_ds = kwarg_ds[0]

    def compute(self, ds):
        if self._is_xskillscore:
            ds = self._rechunk(ds)
        kwarg_ds = {self._kwarg_ds: ds}
        return self._func(**kwarg_ds)

    @property
    def is_decomposable(self):
        """True if the metric config opted into a fused-engine kernel."""
        return self._kernel_name is not None

    @property
    def kernel_name(self):
        return self._kernel_name

    @property
    def finalize_name(self):
        return self._finalize_name

    @property
    def kernel(self):
        """The kernel callable, or ``None`` if not opted in.

        Raises ``KeyError`` if the configured name is unknown.
        """
        if self._kernel_name is None:
            return None
        fn = get_kernel(self._kernel_name)
        if fn is None:
            raise KeyError(
                f"metric {self.name!r}: unknown kernel "
                f"{self._kernel_name!r} (known: {sorted(_KERNEL_REGISTRY)})"
            )
        return fn

    @property
    def finalize(self):
        """The finalizer callable, or ``None`` if not opted in.

        Raises ``KeyError`` if the configured name is unknown.
        """
        if self._kernel_name is None:
            return None
        fn = get_finalize(self._finalize_name)
        if fn is None:
            raise KeyError(
                f"metric {self.name!r}: unknown finalize "
                f"{self._finalize_name!r} (known: {sorted(_FINALIZE_REGISTRY)})"
            )
        return fn

    def _rechunk(self, ds):
        if self._dim is None:
            return ds
        dim = [self._dim] if isinstance(self._dim, str) else self._dim
        dim_other = [d for d in ds.dims if d not in dim]
        chunks = {d: -1 for d in dim}
        for d in dim_other:
            chunks[d] = 1
        return ds.chunk(chunks)


def verify(fcst, obs, func_path, inputs, **kwargs):
    func = _resolve_function(func_path=func_path)
    datasets = {
        "forecast": fcst,
        "observation": obs,
    }
    input_kwargs = {arg_name: datasets[ds_type] for arg_name, ds_type in inputs.items()}

    all_kwargs = {**input_kwargs, **kwargs}

    result = func(**all_kwargs)
    return result
