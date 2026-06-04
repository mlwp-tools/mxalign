"""Fused verification engine (Phase 2 / lever B, recipe-based).

For each reference_time, one client.submit task:

  1. Loads the per-rt slice directly from the underlying store
     (NetCDF per rt for forecasts; zarr region read for ERA5)
     via the loader's `fast_slice_recipe`, bypassing xarray's lazy graphs.
  2. Replays the registered transformations on that small per-rt Dataset.
  3. Applies a sum-decomposable kernel (e.g. squared error for MSE).
  4. Returns numpy partials + per-stage timings.

Driver runs an `as_completed` loop with a bounded submission window
("backpressure"), accumulating partials in driver memory (~few GB total).
After all leaves complete it finalises (e.g. divides by N_rt for means) and
wraps the result into an xr.Dataset matching the legacy engine shape.

Scope (v1):
  - Sum-decomposable metrics with `reduce_dims` containing 'reference_time':
    MSE, MAE, bias (mean error), mean(reference), mean(forecast).
  - Loaders: anemoi-inference (per-rt NetCDF), anemoi-datasets (single zarr).
  - Transformations: rename, kelvin_to_celcius, uv_to_speed (extend by
    adding entries to `_TRANSFORM_IO`).

Validation failures (unsupported metric/transform/loader, missing
reduce_dims, missing fast_slice_recipe) raise immediately. No silent
fallback.
"""
from __future__ import annotations

import logging
import statistics
import threading
import time
import warnings
from collections import deque
from typing import Any, Callable

import numpy as np
import xarray as xr

LOG = logging.getLogger("mxalign")


# ---------------------------------------------------------------------------
# Metric kernels
# ---------------------------------------------------------------------------
# Each kernel takes (fcst, ref) numpy arrays of shape (n_var, n_lt, n_grid)
# and returns a per-sample partial of the same shape that is **summable**
# across reference_time. The finalize step (mean = sum / N, sum = sum)
# is applied after all leaves are reduced.

def _kernel_squared_error(fcst: np.ndarray, ref: np.ndarray) -> np.ndarray:
    diff = fcst.astype(np.float32, copy=False) - ref.astype(np.float32, copy=False)
    return diff * diff


def _kernel_abs_error(fcst: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(
        fcst.astype(np.float32, copy=False) - ref.astype(np.float32, copy=False)
    )


def _kernel_error(fcst: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return fcst.astype(np.float32, copy=False) - ref.astype(np.float32, copy=False)


def _kernel_identity_fcst(fcst: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return fcst.astype(np.float32, copy=False)


def _kernel_identity_ref(fcst: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return ref.astype(np.float32, copy=False)


# func_path -> (kernel, finalize_kind in {"mean", "sum"})
_FUSED_KERNELS: dict[str, tuple[Callable, str]] = {
    "scores.continuous.mse": (_kernel_squared_error, "mean"),
    "scores.continuous.mae": (_kernel_abs_error, "mean"),
    "scores.continuous.bias": (_kernel_error, "mean"),
    "scores.continuous.mean_error": (_kernel_error, "mean"),
}


# ---------------------------------------------------------------------------
# Transformation source-variable bookkeeping
# ---------------------------------------------------------------------------
# Each entry returns (inputs, outputs) variable lists given the transform's
# kwargs (as recorded by Runner.transform_datasets). Used to walk the
# transformation chain backwards from `common_vars` to "what to load from
# the source".

def _io_uv_to_speed(kwargs):
    u = kwargs["u"]; v = kwargs["v"]; s = kwargs["speed"]
    u = [u] if isinstance(u, str) else list(u)
    v = [v] if isinstance(v, str) else list(v)
    s = [s] if isinstance(s, str) else list(s)
    return u + v, s


def _io_kelvin_to_celcius(kwargs):
    v = kwargs["variables"]
    v = [v] if isinstance(v, str) else list(v)
    return v, v  # in-place


def _io_rename(kwargs):
    d = kwargs["rename_dict"]  # new_name -> old_name(s)
    outputs = list(d.keys())
    inputs: list[str] = []
    for v in d.values():
        inputs.extend(v if isinstance(v, list) else [v])
    return inputs, outputs


_TRANSFORM_IO: dict[str, Callable] = {
    "uv_to_speed": _io_uv_to_speed,
    "kelvin_to_celcius": _io_kelvin_to_celcius,
    "rename": _io_rename,
}


def _derive_source_vars(common_vars, transforms_for_ds):
    """Walk transformations backwards to derive the set of source variables
    that need to be read from the store for one dataset."""
    needed = set(common_vars)
    for tname, tkwargs in reversed(transforms_for_ds):
        if tname not in _TRANSFORM_IO:
            raise NotImplementedError(
                f"fused engine: transformation {tname!r} has no input/output "
                f"spec in _TRANSFORM_IO; add one or use engine=xarray"
            )
        inputs, outputs = _TRANSFORM_IO[tname](tkwargs)
        if any(o in needed for o in outputs):
            needed -= set(outputs)
            needed |= set(inputs)
    return sorted(needed)


# ---------------------------------------------------------------------------
# Per-rt slice loaders (worker-side)
# ---------------------------------------------------------------------------

def _rt_key(rt) -> int:
    """Canonical hashable key for a reference_time: ns-since-epoch int."""
    return int(np.datetime64(rt, "ns").astype("int64"))


def _load_slice(recipe, rt_value, lead_times, var_names) -> xr.Dataset:
    kind = recipe["kind"]
    if kind == "anemoi-inference-nc":
        return _load_anemoi_inference_slice(recipe, rt_value, lead_times, var_names)
    if kind == "anemoi-datasets-zarr":
        return _load_anemoi_datasets_slice(recipe, rt_value, lead_times, var_names)
    raise NotImplementedError(f"fused engine: unknown recipe kind {kind!r}")


def _load_anemoi_inference_slice(recipe, rt_value, lead_times, var_names) -> xr.Dataset:
    path = recipe["files_by_rt"][_rt_key(rt_value)]
    engine = recipe["engine"]
    with xr.open_dataset(path, engine=engine) as src:
        # Subset variables + lead_times *lazily* and only then call .load().
        # Doing .load() up front (the previous behaviour) forces a read of the
        # full time axis even when the file holds more steps than we need; it
        # also turns the time-axis selection into an in-memory fancy index
        # instead of a hyperslab read. Selecting first lets HDF5 issue a
        # single contiguous read for the steady-state (cadence-1) case.
        sub = src[list(var_names)]
        if "time" in sub.dims:
            times = sub["time"].values
            lts = (times - times[0]).astype("timedelta64[ns]")
            sub = sub.assign_coords({"lead_time": ("time", lts)}).swap_dims(
                {"time": "lead_time"}
            )
        if "values" in sub.dims:
            sub = sub.rename_dims({"values": "grid_index"})
        requested = np.asarray(
            [np.timedelta64(int(lt), "ns") for lt in lead_times],
            dtype="timedelta64[ns]",
        )
        file_lts = sub["lead_time"].values.astype("timedelta64[ns]")
        pos = np.searchsorted(file_lts, requested)
        if pos.max() >= file_lts.size or not np.all(file_lts[pos] == requested):
            bad = requested[
                (pos >= file_lts.size)
                | (file_lts[pos.clip(max=file_lts.size - 1)] != requested)
            ]
            raise ValueError(
                f"fused engine: missing lead_times in {path}: {bad[:5]}... "
                f"(reference_time={rt_value})"
            )
        # Contiguous fast path → hyperslab; otherwise fancy index.
        pos_arr = np.asarray(pos)
        if pos_arr.size == 0:
            contiguous = False
        elif pos_arr.size == 1:
            contiguous = True
        else:
            contiguous = bool(np.all(np.diff(pos_arr) == 1))
        if contiguous:
            sub = sub.isel(
                lead_time=slice(int(pos_arr[0]), int(pos_arr[-1]) + 1)
            )
        else:
            sub = sub.isel(lead_time=xr.DataArray(pos_arr, dims="lead_time"))
        ds = sub.load()
    return ds


def _load_anemoi_datasets_slice(recipe, rt_value, lead_times, var_names) -> xr.Dataset:
    path = recipe["path"]
    src = xr.open_zarr(path, consolidated=recipe.get("consolidated", False))

    # 'dates' coord on the 'time' dim is the canonical valid_time array.
    valid_times = src["dates"].astype("datetime64[ns]").load().values
    var_attr = list(src.attrs["variables"])
    try:
        var_idx = np.array([var_attr.index(v) for v in var_names], dtype=np.int64)
    except ValueError as e:
        raise ValueError(
            f"fused engine: variable not found in {path}: {e}"
        ) from None

    rt = np.datetime64(rt_value, "ns")
    requested_vts = np.array(
        [rt + np.timedelta64(lt, "ns") for lt in lead_times], dtype="datetime64[ns]"
    )
    pos = np.searchsorted(valid_times, requested_vts)
    if pos.max() >= valid_times.size or not np.all(valid_times[pos] == requested_vts):
        bad = requested_vts[
            (pos >= valid_times.size) | (valid_times[pos.clip(max=valid_times.size - 1)] != requested_vts)
        ]
        raise ValueError(
            f"fused engine: missing valid_times in {path}: {bad[:5]}... "
            f"(reference_time={rt_value})"
        )

    arr = src["data"].isel(ensemble=0)
    # If `pos` is strictly contiguous (the common case: 1 h cadence lead_times),
    # issue a single slice read instead of a fancy index. Fancy indexing along
    # `time` triggers one chunk read per requested step per variable, which on
    # finely-time-chunked zarrs blows up into thousands of small reads. A slice
    # is a single contiguous request and avoids that amplification entirely.
    pos_arr = np.asarray(pos)
    if pos_arr.size == 0:
        contiguous = False
    elif pos_arr.size == 1:
        contiguous = True
    else:
        contiguous = bool(np.all(np.diff(pos_arr) == 1))
    if contiguous:
        start = int(pos_arr[0])
        stop = int(pos_arr[-1]) + 1
        arr_sel = arr.isel(
            time=slice(start, stop),
            variable=xr.DataArray(var_idx, dims="variable_out"),
        )
    else:
        arr_sel = arr.isel(
            time=xr.DataArray(pos_arr, dims="lead_time"),
            variable=xr.DataArray(var_idx, dims="variable_out"),
        )
    loaded = arr_sel.load()
    vals = np.asarray(loaded.values)  # (n_lt, n_var, n_grid)
    # `vals` lead_time axis matches the slice/index we asked for; for the
    # contiguous-slice path it's already in the right order (and length).
    ds = xr.Dataset(
        {
            v: (("lead_time", "grid_index"), vals[:, i, :])
            for i, v in enumerate(var_names)
        }
    )
    return ds


# ---------------------------------------------------------------------------
# Leaf task (runs on worker)
# ---------------------------------------------------------------------------

def _leaf(
    rt_value,
    lead_times,
    common_vars,
    ref_name,
    model_names,
    recipes_by_ds,
    source_vars_by_ds,
    transforms_by_ds,
    metric_kernels,
):
    """One per-reference_time task.

    Returns:
      {
        "rt_value": rt_value,
        "timings": {load_<ds>: float, transform_<ds>: float, kernel: float, total: float},
        "partials": {model_name: {metric_name: np.ndarray(n_var, n_lt, n_grid)}},
      }
    """
    from mxalign.transformations.registry import get_transformation

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    # 1. Load per-dataset slices.
    slices: dict[str, xr.Dataset] = {}
    for ds_name, recipe in recipes_by_ds.items():
        t = time.perf_counter()
        slices[ds_name] = _load_slice(
            recipe, rt_value, lead_times, source_vars_by_ds[ds_name]
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

    # 4. Apply kernels.
    ref = arrays[ref_name]
    partials: dict[str, dict[str, np.ndarray]] = {}
    t = time.perf_counter()
    for m in model_names:
        fcst = arrays[m]
        partials[m] = {
            mn: kernel(fcst, ref) for mn, (kernel, _) in metric_kernels.items()
        }
    timings["kernel"] = time.perf_counter() - t

    timings["total"] = time.perf_counter() - t0
    return {"rt_value": rt_value, "timings": timings, "partials": partials}


def _leaf_bundled(rt_value, static):
    """Worker-side trampoline: unpack the scattered static bundle and call _leaf.

    `static` is a plain dict that was shipped to every worker once via
    `client.scatter(..., broadcast=True)`. Dask resolves the Future to its
    materialized value before invoking this function.
    """
    return _leaf(
        rt_value,
        static["lead_times_ns"],
        common_vars=static["common_vars"],
        ref_name=static["ref_name"],
        model_names=static["model_names"],
        recipes_by_ds=static["recipes_by_ds"],
        source_vars_by_ds=static["source_vars_by_ds"],
        transforms_by_ds=static["transforms_by_ds"],
        metric_kernels=static["metric_kernels"],
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _validate(reference, datasets, loaders, transforms_by_ds, metrics_cfg, ref_name):
    # 1. Every dataset must have a recipe.
    recipes: dict[str, dict] = {}
    for name, loader in loaders.items():
        if not hasattr(loader, "fast_slice_recipe"):
            raise NotImplementedError(
                f"fused engine: loader for dataset {name!r} has no "
                f"fast_slice_recipe(); use engine=xarray or extend the loader."
            )
        recipe = loader.fast_slice_recipe()
        if recipe is None:
            raise NotImplementedError(
                f"fused engine: loader {type(loader).__name__!r} declined to "
                f"produce a fast-slice recipe for dataset {name!r} (e.g. "
                f"unsupported file layout); use engine=xarray."
            )
        recipes[name] = recipe

    # 2. Every metric must be in the allow-list with reduce_dims=[reference_time].
    metric_kernels: dict[str, tuple[Callable, str]] = {}
    for mn, mcfg in metrics_cfg.items():
        func_path = mcfg.get("function")
        if func_path not in _FUSED_KERNELS:
            raise NotImplementedError(
                f"fused engine: metric {mn!r} uses function {func_path!r} which "
                f"is not in the sum-decomposable allow-list "
                f"({sorted(_FUSED_KERNELS)}); use engine=xarray."
            )
        rd = mcfg.get("reduce_dims") or []
        rd = [rd] if isinstance(rd, str) else list(rd)
        if "reference_time" not in rd:
            raise ValueError(
                f"fused engine: metric {mn!r} has reduce_dims={rd}; the fused "
                f"engine requires 'reference_time' among reduce_dims."
            )
        metric_kernels[mn] = _FUSED_KERNELS[func_path]

    # 3. Reference must be one of the datasets.
    if ref_name not in datasets:
        raise ValueError(f"fused engine: reference {ref_name!r} not in datasets")

    return recipes, metric_kernels


def _make_xr_result(accums, finalizers, n_rt, common_vars, reference, model_order,
                    metric_order):
    """Wrap accumulated partials into an xr.Dataset matching the legacy shape:
        dims = (model, metric, variable, lead_time, grid_index)
    Coords: model, metric, variable, lead_time (+ latitude/longitude on grid_index).
    """
    lead_time = reference["lead_time"].values
    lat = reference["latitude"].values if "latitude" in reference.coords else None
    lon = reference["longitude"].values if "longitude" in reference.coords else None

    # (model, metric, variable, lead_time, grid_index)
    arr_by_metric: dict[str, np.ndarray] = {}
    for mn in metric_order:
        finalize = finalizers[mn]
        stacked = np.stack(
            [
                accums[m][mn] / (n_rt if finalize == "mean" else 1)
                for m in model_order
            ],
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
        # Collect per-stage timings across the window.
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


def _prefetch_nc_file(path: str) -> None:
    """Read *path* sequentially in a daemon thread to populate the OS page
    cache. Errors are silently swallowed — a failed prefetch just means the
    next leaf reads cold, which is no worse than before."""
    try:
        with open(path, "rb") as fh:
            buf = bytearray(8 << 20)  # 8 MB read buffer
            while fh.readinto(buf):
                pass
    except OSError:
        pass


def _schedule_prefetch(
    rt_values,
    idx: int,
    recipes: dict,
    prefetch_ahead: int,
) -> None:
    """Start a background prefetch daemon thread for the forecast NC file(s)
    belonging to rt_values[idx + prefetch_ahead], if any.
    Only fires for 'anemoi-inference-nc' recipes (not zarr).
    """
    target_idx = idx + prefetch_ahead
    if target_idx >= len(rt_values):
        return
    rt = rt_values[target_idx]
    for name, recipe in recipes.items():
        if recipe.get("kind") != "anemoi-inference-nc":
            continue
        key = _rt_key(rt)
        path = recipe.get("files_by_rt", {}).get(key)
        if path:
            threading.Thread(
                target=_prefetch_nc_file,
                args=(path,),
                daemon=True,
                name=f"mxalign-prefetch-{name}-{target_idx}",
            ).start()


def compute_metrics_fused(
    datasets,
    loaders,
    transforms_by_ds,
    reference_name,
    common_vars,
    metrics_cfg,
    engine_cfg,
):
    """Driver entry point. Returns an xr.Dataset shaped
    (model, metric, variable, lead_time, grid_index)."""
    common_vars = sorted(common_vars)
    reference = datasets[reference_name]
    model_order = sorted(n for n in datasets if n != reference_name)
    metric_order = list(metrics_cfg.keys())

    recipes, metric_kernels = _validate(
        reference, datasets, loaders, transforms_by_ds, metrics_cfg, reference_name
    )

    # Derive per-dataset source variables (walk transformations backwards).
    source_vars_by_ds = {
        name: _derive_source_vars(common_vars, transforms_by_ds.get(name, []))
        for name in datasets
    }

    # Per-rt iteration: drive from the reference dataset's reference_time.
    if "reference_time" not in reference.dims:
        raise ValueError(
            "fused engine: reference dataset has no 'reference_time' dim; "
            "this engine requires forecast-shaped reference."
        )
    rt_values = reference["reference_time"].values
    lead_times = reference["lead_time"].values  # timedelta64[ns]
    # Convert lead_times to integer ns for stable pickling.
    lead_times_ns = [int(np.timedelta64(lt, "ns").astype("int64")) for lt in lead_times]

    n_rt = len(rt_values)
    finalizers = {mn: kind for mn, (_, kind) in metric_kernels.items()}

    # Pre-allocate driver-side accumulators (one per model+metric, ~1.5GB each).
    n_var = len(common_vars)
    n_lt = len(lead_times)
    # We let the first arriving partial allocate via copy; saves a guess at n_grid.
    accums: dict[str, dict[str, np.ndarray | None]] = {
        m: {mn: None for mn in metric_order} for m in model_order
    }

    # Try to get a Client; if none, run serial in-process.
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

    # Prefetch: background daemon threads warm the OS page cache for the next
    # NC file(s) while the current leaf is being processed. Enabled via
    # `prefetch: true` in the `verification:` yaml block. Only fires for
    # anemoi-inference-nc recipes; zarr datasets are skipped.
    prefetch_enabled = bool(engine_cfg.get("prefetch", False))
    # Look-ahead depth: start prefetching the file for leaf N+prefetch_ahead
    # when leaf N is submitted/consumed. Default max_in_flight+1.
    prefetch_ahead = max(1, int(engine_cfg.get("prefetch_ahead", max_in_flight + 1)))

    LOG.info(
        "[mxalign] fused start n_rt=%d n_models=%d n_metrics=%d n_vars=%d "
        "n_lt=%d max_in_flight=%d client=%s recipes={%s}",
        n_rt, len(model_order), len(metric_order), n_var, n_lt, max_in_flight,
        "yes" if client is not None else "no (serial)",
        ", ".join(f"{n}:{r['kind']}" for n, r in recipes.items()),
    )

    timings_window: deque = deque(maxlen=64)
    last_progress_log = time.perf_counter()
    last_completion = time.perf_counter()
    t_start = time.perf_counter()
    done = 0

    def _consume(result):
        nonlocal done, last_completion
        partials = result["partials"]
        for m, per_metric in partials.items():
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
        recipes_by_ds=recipes,
        source_vars_by_ds=source_vars_by_ds,
        transforms_by_ds=transforms_by_ds,
        metric_kernels=metric_kernels,
    )

    if client is None:
        # Serial fallback (mainly for --cluster threads).
        for i, rt in enumerate(rt_values):
            if prefetch_enabled:
                _schedule_prefetch(rt_values, i, recipes, prefetch_ahead)
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
        # Scatter the (large, identical-per-submit) static payload once and
        # broadcast it to all workers. Each subsequent client.submit then ships
        # only the per-leaf rt + lead_times + a pointer to the scattered
        # bundle, keeping the per-submit graph size in the KB range.
        # `lead_times_ns` is small (<=145 ints) but we scatter it too for
        # symmetry. Broadcast=True ensures it's already on every worker before
        # the first submit, so workers never pull from the scheduler at task
        # start.
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
        # Prime the window (and optionally prime the prefetch pipeline).
        for _ in range(min(max_in_flight, n_rt)):
            if prefetch_enabled:
                _schedule_prefetch(rt_values, i_next, recipes, prefetch_ahead)
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
                    _schedule_prefetch(rt_values, i_next, recipes, prefetch_ahead)
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
        accums, finalizers, n_rt, common_vars, reference, model_order, metric_order
    )
