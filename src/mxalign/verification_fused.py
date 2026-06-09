"""Fused verification engine.

For each reference_time, one client.submit task:

  1. Loads the per-rt slice directly from the underlying store via
     ``loader.slice(rt, lead_times, source_vars)`` — bypassing xarray's
     lazy graphs.
  2. Replays the recorded transformations on that small per-rt Dataset.
  3. Applies a sum-decomposable kernel (e.g. squared error for MSE).
  4. Returns numpy partials + per-stage timings.

Driver runs an ``as_completed`` loop with a bounded submission window
("backpressure"), accumulating partials in driver memory (~few GB total).
After all leaves complete it finalises (e.g. divides by N_rt for means) and
wraps the result into an ``xr.Dataset`` matching the legacy engine shape.

Required abstractions (extend these to add capability):

  * Loaders override :meth:`mxalign.loaders.base.BaseLoader.slice` for the
    per-rt fast read. May optionally provide ``prefetch_path(rt)`` to
    enable OS-page-cache prefetch.
  * Metric functions are decorated with
    :func:`mxalign.verification.fused_metric` to expose a numpy kernel +
    finalizer. The bundled :mod:`mxalign.scores` ships
    ``mse`` / ``mae`` / ``bias`` / ``mean_error``.
  * Transformations declare an I/O signature via
    ``register_transformation(..., signature=...)`` so the engine can
    derive which source variables to load.

Validation failures raise immediately with a message pointing at the
abstraction to extend; no silent fallback.
"""
from __future__ import annotations

import inspect
import logging
import statistics
import threading
import time
import warnings
from collections import deque

import numpy as np
import xarray as xr

from .loaders.base import BaseLoader
from .transformations.external import _resolve_function
from .transformations.registry import get_signature, get_transformation
from .verification import get_fused_kernel

LOG = logging.getLogger("mxalign")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rt_key(rt) -> int:
    """Canonical hashable key for a reference_time: ns-since-epoch int."""
    return int(np.datetime64(rt, "ns").astype("int64"))


def _derive_source_vars(common_vars, transforms_for_ds):
    """Walk transformations backwards (using their declared signatures) to
    determine which source variables must be read from the store for one
    dataset."""
    needed = set(common_vars)
    for tname, tkwargs in reversed(transforms_for_ds):
        sig = get_signature(tname)
        if sig is None:
            raise NotImplementedError(
                f"engine=fused: transformation {tname!r} has no declared I/O "
                f"signature; add `signature=...` to its "
                f"`register_transformation(...)` call or use engine=xarray."
            )
        inputs, outputs = sig(**tkwargs)
        if any(o in needed for o in outputs):
            needed -= set(outputs)
            needed |= set(inputs)
    return sorted(needed)


# ---------------------------------------------------------------------------
# Leaf task (runs on worker)
# ---------------------------------------------------------------------------

def _leaf(
    rt_value,
    lead_times_ns,
    common_vars,
    ref_name,
    model_names,
    loaders,
    source_vars_by_ds,
    transforms_by_ds,
    metric_specs,  # {metric_name: (kernel_callable, inputs_or_None)}
):
    """One per-reference_time task.

    Returns:
      {
        "rt_value": rt_value,
        "timings": {load_<ds>: float, transform_<ds>: float, kernel: float, total: float},
        "partials": {model_name: {metric_name: np.ndarray(n_var, n_lt, n_grid)}},
      }
    """
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    # 1. Load per-dataset slices via the loader's eager slice() method.
    slices: dict[str, xr.Dataset] = {}
    for ds_name, loader in loaders.items():
        t = time.perf_counter()
        slices[ds_name] = loader.slice(
            rt_value, lead_times_ns, source_vars_by_ds[ds_name]
        )
        timings[f"load_{ds_name}"] = time.perf_counter() - t

    # 2. Replay transformations in recorded order.
    for ds_name, ds in list(slices.items()):
        t = time.perf_counter()
        for tname, tkwargs in transforms_by_ds.get(ds_name, []):
            func = get_transformation(tname)
            ds = func(ds.copy(), **tkwargs)
        slices[ds_name] = ds
        timings[f"transform_{ds_name}"] = time.perf_counter() - t

    # 3. Stack to canonical (n_var, n_lt, n_grid) float32 numpy.
    arrays: dict[str, np.ndarray] = {}
    for ds_name, ds in slices.items():
        arrays[ds_name] = np.stack(
            [
                np.ascontiguousarray(ds[v].values, dtype=np.float32)
                for v in common_vars
            ],
            axis=0,
        )

    # 4. Apply kernels. Each metric binds its declared `inputs:` roles to the
    #    kernel's parameters by name: role 'reference' -> reference array,
    #    any other role -> the model array being scored. Metrics without an
    #    `inputs:` block fall back to positional (forecast, reference).
    ref = arrays[ref_name]
    partials: dict[str, dict[str, np.ndarray]] = {}
    t = time.perf_counter()
    for m in model_names:
        model_arr = arrays[m]
        out: dict[str, np.ndarray] = {}
        for mn, (kern, inputs) in metric_specs.items():
            if inputs:
                kwargs = {
                    arg: (ref if role == "reference" else model_arr)
                    for arg, role in inputs.items()
                }
                out[mn] = kern(**kwargs)
            else:
                out[mn] = kern(model_arr, ref)
        partials[m] = out
    timings["kernel"] = time.perf_counter() - t

    timings["total"] = time.perf_counter() - t0
    return {"rt_value": rt_value, "timings": timings, "partials": partials}


def _leaf_bundled(rt_value, static):
    """Worker-side trampoline: unpack the scattered static bundle and call
    :func:`_leaf`.

    ``static`` is a plain dict that was shipped to every worker once via
    ``client.scatter(..., broadcast=True)``. Dask resolves the Future to
    its materialized value before invoking this function.
    """
    return _leaf(
        rt_value,
        static["lead_times_ns"],
        common_vars=static["common_vars"],
        ref_name=static["ref_name"],
        model_names=static["model_names"],
        loaders=static["loaders"],
        source_vars_by_ds=static["source_vars_by_ds"],
        transforms_by_ds=static["transforms_by_ds"],
        metric_specs=static["metric_specs"],
    )


# ---------------------------------------------------------------------------
# Driver: validation
# ---------------------------------------------------------------------------

def _resolve_kernel_inputs(metric_name, func_path, kernel, inputs):
    """Validate a metric's YAML ``inputs:`` map against its fused kernel.

    ``inputs`` maps the metric function's argument names to roles
    (``reference`` for the reference dataset, any other role for the model
    being scored). The fused leaf binds these names directly onto the
    kernel, so the kernel must accept them. Returns the inputs dict (copy),
    or ``None`` when no ``inputs:`` block was given (positional fallback).
    """
    if not inputs:
        return None
    roles = list(inputs.values())
    n_ref = sum(1 for r in roles if r == "reference")
    if n_ref != 1:
        raise ValueError(
            f"engine=fused: metric {metric_name!r} inputs={inputs} must map "
            f"exactly one argument to role 'reference' (got {n_ref})."
        )
    sig = inspect.signature(kernel)
    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not has_var_kw:
        valid = {
            p.name
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        bad = [a for a in inputs if a not in valid]
        if bad:
            raise ValueError(
                f"engine=fused: metric {metric_name!r} function {func_path!r} "
                f"fused kernel does not accept input argument(s) {bad}; kernel "
                f"parameters are {sorted(valid)}."
            )
    return dict(inputs)


def _validate(reference, datasets, loaders, metrics_cfg, ref_name):
    # 1. Reference must be one of the datasets and forecast-shaped.
    if ref_name not in datasets:
        raise ValueError(f"engine=fused: reference {ref_name!r} not in datasets")
    if "reference_time" not in reference.dims:
        raise ValueError(
            "engine=fused: reference dataset has no 'reference_time' dim; "
            "this engine requires forecast-shaped reference."
        )

    # 2. Every loader must override BaseLoader.slice.
    for name, loader in loaders.items():
        if type(loader).slice is BaseLoader.slice:
            raise NotImplementedError(
                f"engine=fused: loader {type(loader).__name__!r} for dataset "
                f"{name!r} does not override BaseLoader.slice(); add a "
                f"slice() method or use engine=xarray."
            )

    # 3. Every metric must resolve to a function with a fused fast path,
    #    and must include 'reference_time' among its reduce_dims.
    metric_specs: dict[str, tuple] = {}
    metric_finalizers: dict[str, "callable"] = {}
    for mn, mcfg in metrics_cfg.items():
        func_path = mcfg.get("function")
        if not func_path:
            raise ValueError(
                f"engine=fused: metric {mn!r} has no 'function:' entry."
            )
        fn = _resolve_function(func_path)
        fast = get_fused_kernel(fn)
        if fast is None:
            raise NotImplementedError(
                f"engine=fused: metric {mn!r} uses function {func_path!r} "
                f"which has no fused fast path. Decorate the function with "
                f"@fused_metric or use one of mxalign.scores.* "
                f"(e.g. mxalign.scores.mse)."
            )
        kernel, finalize = fast
        rd = mcfg.get("reduce_dims") or []
        rd = [rd] if isinstance(rd, str) else list(rd)
        if "reference_time" not in rd:
            raise ValueError(
                f"engine=fused: metric {mn!r} has reduce_dims={rd}; the fused "
                f"engine requires 'reference_time' among reduce_dims."
            )
        inputs = _resolve_kernel_inputs(mn, func_path, kernel, mcfg.get("inputs"))
        metric_specs[mn] = (kernel, inputs)
        metric_finalizers[mn] = finalize

    return metric_specs, metric_finalizers


# ---------------------------------------------------------------------------
# Driver: result wrap
# ---------------------------------------------------------------------------

def _make_xr_result(accums, finalizers, n_rt, common_vars, reference,
                    model_order, metric_order):
    """Wrap accumulated partials into an ``xr.Dataset`` matching the legacy
    shape: dims = ``(model, metric, variable, lead_time, grid_index)``.

    Coords: ``model``, ``metric``, ``variable``, ``lead_time``
    (+ ``latitude`` / ``longitude`` on ``grid_index`` when present on the
    reference).
    """
    lead_time = reference["lead_time"].values
    lat = reference["latitude"].values if "latitude" in reference.coords else None
    lon = reference["longitude"].values if "longitude" in reference.coords else None

    arr_by_metric: dict[str, np.ndarray] = {}
    for mn in metric_order:
        finalize = finalizers[mn]
        stacked = np.stack(
            [finalize(accums[m][mn], n_rt) for m in model_order],
            axis=0,
        )  # (n_model, n_var, n_lt, n_grid)
        arr_by_metric[mn] = stacked

    full = np.stack([arr_by_metric[mn] for mn in metric_order], axis=1)
    # full: (n_model, n_metric, n_var, n_lt, n_grid)

    coords = {
        "model": list(model_order),
        "metric": list(metric_order),
        "variable": list(common_vars),
        "lead_time": lead_time,
    }
    if lat is not None:
        coords["latitude"] = ("grid_index", lat)
    if lon is not None:
        coords["longitude"] = ("grid_index", lon)

    return xr.DataArray(
        full,
        dims=("model", "metric", "variable", "lead_time", "grid_index"),
        coords=coords,
    ).to_dataset(name="metrics")


# ---------------------------------------------------------------------------
# Driver: progress + prefetch
# ---------------------------------------------------------------------------

def _log_progress(done, total, t_start, timings_window, in_flight):
    elapsed = time.perf_counter() - t_start
    throughput = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / throughput if throughput > 0 else float("nan")
    parts = [
        f"[mxalign] fused progress done={done}/{total}",
        f"inflight={in_flight}",
        f"elapsed={elapsed:.1f}s",
        f"throughput={throughput:.2f}leaf/s",
        f"eta={eta:.0f}s",
    ]
    if timings_window:
        keys = set().union(*(t.keys() for t in timings_window))
        bits = []
        for k in sorted(keys):
            vals = [t[k] for t in timings_window if k in t]
            if not vals:
                continue
            p50 = statistics.median(vals)
            p95 = sorted(vals)[max(0, int(0.95 * len(vals)) - 1)]
            bits.append(f"{k}(p50={p50:.2f}s,p95={p95:.2f}s)")
        parts.append("timings=[" + " ".join(bits) + "]")
    LOG.info(" ".join(parts))


def _prefetch_file(path: str) -> None:
    """Read *path* sequentially in a daemon thread to populate the OS page
    cache. Errors are silently swallowed — a failed prefetch just means
    the next leaf reads cold, which is no worse than before."""
    try:
        with open(path, "rb") as fh:
            buf = bytearray(8 << 20)  # 8 MB read buffer
            while fh.readinto(buf):
                pass
    except OSError:
        pass


def _schedule_prefetch(rt_values, idx: int, loaders: dict,
                       prefetch_ahead: int) -> None:
    """Start a background prefetch daemon thread for each loader's file(s)
    at ``rt_values[idx + prefetch_ahead]``. Loaders that do not expose a
    ``prefetch_path(rt)`` method are skipped (typically zarr-backed
    loaders, where OS-level prefetch is not useful)."""
    target_idx = idx + prefetch_ahead
    if target_idx >= len(rt_values):
        return
    rt = rt_values[target_idx]
    for name, loader in loaders.items():
        get_path = getattr(loader, "prefetch_path", None)
        if get_path is None:
            continue
        path = get_path(rt)
        if path:
            threading.Thread(
                target=_prefetch_file,
                args=(path,),
                daemon=True,
                name=f"mxalign-prefetch-{name}-{target_idx}",
            ).start()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def compute_metrics_fused(
    datasets,
    loaders,
    transforms_by_ds,
    reference_name,
    common_vars,
    metrics_cfg,
    engine_cfg,
):
    """Driver entry point. Returns an ``xr.Dataset`` shaped
    ``(model, metric, variable, lead_time, grid_index)``."""
    common_vars = sorted(common_vars)
    reference = datasets[reference_name]
    model_order = sorted(n for n in datasets if n != reference_name)
    metric_order = list(metrics_cfg.keys())

    metric_specs, metric_finalizers = _validate(
        reference, datasets, loaders, metrics_cfg, reference_name
    )

    # Per-dataset source variables (walk transformation signatures backwards).
    source_vars_by_ds = {
        name: _derive_source_vars(common_vars, transforms_by_ds.get(name, []))
        for name in datasets
    }

    rt_values = reference["reference_time"].values
    lead_times = reference["lead_time"].values  # timedelta64[ns]
    # Convert lead_times to integer ns for stable pickling.
    lead_times_ns = [int(np.timedelta64(lt, "ns").astype("int64")) for lt in lead_times]

    n_rt = len(rt_values)
    n_var = len(common_vars)
    n_lt = len(lead_times)
    # Driver-side accumulators (one per model+metric). The first arriving
    # partial allocates via copy — avoids guessing n_grid up front.
    accums: dict[str, dict[str, np.ndarray | None]] = {
        m: {mn: None for mn in metric_order} for m in model_order
    }

    # Optional dask client.
    client = None
    try:
        from dask.distributed import default_client, as_completed
        client = default_client()
    except Exception:
        client = None

    max_in_flight_cfg = engine_cfg.get("max_in_flight")
    if client is not None:
        n_workers = max(1, len(client.scheduler_info().get("workers", {})))
        default_window = 2 * n_workers
        max_in_flight = int(max_in_flight_cfg) if max_in_flight_cfg else default_window
    else:
        max_in_flight = 1

    # Prefetch: background daemon threads warm the OS page cache for the
    # next forecast file(s) while the current leaf is being processed.
    # Enabled via `prefetch: true` in the `verification:` yaml block.
    prefetch_enabled = bool(engine_cfg.get("prefetch", False))
    prefetch_ahead = max(1, int(engine_cfg.get("prefetch_ahead", max_in_flight + 1)))

    LOG.info(
        "[mxalign] fused start n_rt=%d n_models=%d n_metrics=%d n_vars=%d "
        "n_lt=%d max_in_flight=%d client=%s loaders={%s}",
        n_rt, len(model_order), len(metric_order), n_var, n_lt, max_in_flight,
        "yes" if client is not None else "no (serial)",
        ", ".join(f"{n}:{type(l).__name__}" for n, l in loaders.items()),
    )

    timings_window: deque = deque(maxlen=64)
    last_progress_log = time.perf_counter()
    last_completion = time.perf_counter()
    t_start = time.perf_counter()
    done = 0

    def _consume(result):
        nonlocal done, last_completion
        for m, per_metric in result["partials"].items():
            for mn, arr in per_metric.items():
                if accums[m][mn] is None:
                    accums[m][mn] = arr  # take ownership
                else:
                    accums[m][mn] += arr
        timings_window.append(result["timings"])
        done += 1
        last_completion = time.perf_counter()

    leaf_kwargs = dict(
        common_vars=common_vars,
        ref_name=reference_name,
        model_names=model_order,
        loaders=loaders,
        source_vars_by_ds=source_vars_by_ds,
        transforms_by_ds=transforms_by_ds,
        metric_specs=metric_specs,
    )

    if client is None:
        # Serial fallback (mainly for --cluster threads).
        for i, rt in enumerate(rt_values):
            if prefetch_enabled:
                _schedule_prefetch(rt_values, i, loaders, prefetch_ahead)
            try:
                result = _leaf(rt, lead_times_ns, **leaf_kwargs)
            except Exception:
                LOG.exception("[mxalign] fused leaf-failed rt_idx=%d rt=%s", i, rt)
                raise
            _consume(result)
            now = time.perf_counter()
            if now - last_progress_log >= 15.0:
                _log_progress(done, n_rt, t_start, list(timings_window), 0)
                last_progress_log = now
    else:
        # Scatter the (identical-per-submit) static payload once and broadcast
        # to all workers. Each subsequent client.submit then ships only the
        # per-leaf rt + a pointer to the scattered bundle, keeping the
        # per-submit graph size in the KB range.
        static_bundle = dict(leaf_kwargs)
        static_bundle["lead_times_ns"] = lead_times_ns
        static_future = client.scatter(static_bundle, broadcast=True, hash=False)

        # Suppress the (now-spurious) per-submit "Sending large graph" warning;
        # with the scattered bundle each submit ships only ~hundreds of bytes.
        warnings.filterwarnings(
            "ignore",
            message="Sending large graph of size",
            category=UserWarning,
            module=r"distributed\.client",
        )

        # Streaming as_completed with a sliding submission window.
        ac = as_completed()
        i_next = 0
        in_flight = 0
        for _ in range(min(max_in_flight, n_rt)):
            if prefetch_enabled:
                _schedule_prefetch(rt_values, i_next, loaders, prefetch_ahead)
            fut = client.submit(_leaf_bundled, rt_values[i_next], static_future,
                                pure=False)
            fut._mxalign_rt_idx = i_next  # informational
            ac.add(fut)
            i_next += 1
            in_flight += 1
        for fut in ac:
            try:
                result = fut.result()
            except Exception:
                LOG.exception(
                    "[mxalign] fused leaf-failed rt_idx=%d",
                    getattr(fut, "_mxalign_rt_idx", -1),
                )
                raise
            _consume(result)
            in_flight -= 1
            # Release the future (and its scheduler-held result) ASAP.
            try:
                fut.release()
            except Exception:
                pass
            if i_next < n_rt:
                if prefetch_enabled:
                    _schedule_prefetch(rt_values, i_next, loaders, prefetch_ahead)
                fut2 = client.submit(_leaf_bundled, rt_values[i_next],
                                     static_future, pure=False)
                fut2._mxalign_rt_idx = i_next
                ac.add(fut2)
                i_next += 1
                in_flight += 1
            now = time.perf_counter()
            if now - last_progress_log >= 15.0:
                _log_progress(done, n_rt, t_start, list(timings_window), in_flight)
                last_progress_log = now
            if now - last_completion >= 60.0 and in_flight > 0:
                LOG.warning(
                    "[mxalign] fused stall: no leaf completion for %.0fs "
                    "(done=%d/%d, inflight=%d)",
                    now - last_completion, done, n_rt, in_flight,
                )
                last_completion = now  # de-spam

    # Final progress line.
    _log_progress(done, n_rt, t_start, list(timings_window), 0)

    return _make_xr_result(
        accums, metric_finalizers, n_rt, common_vars, reference, model_order,
        metric_order,
    )
