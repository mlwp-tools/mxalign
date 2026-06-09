from .transformations.external import _resolve_function
from functools import partial


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
