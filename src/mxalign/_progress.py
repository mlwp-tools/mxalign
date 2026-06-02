"""Lightweight progress + diagnostics helpers for mxalign (Phase 0).

All output goes to the 'mxalign' logger at INFO, single-line key=value
format so it is grep-friendly in SLURM logs.

Helpers degrade silently if dask.distributed / psutil are unavailable.
"""
from __future__ import annotations

import logging
import threading
import time

LOG = logging.getLogger("mxalign")


def count_tasks(obj) -> int:
    """Total task count across a dask-backed xarray/dask collection.

    Sums per-layer counts from the HighLevelGraph; avoids materializing the
    full task dict (which can itself be slow for huge graphs).
    """
    try:
        graph = obj.__dask_graph__()
    except AttributeError:
        return 0
    try:
        return sum(len(layer) for layer in graph.layers.values())
    except AttributeError:
        try:
            return len(dict(graph))
        except Exception:
            return -1


def _get_client():
    try:
        from dask.distributed import default_client
        return default_client()
    except Exception:
        return None


def _worker_rss_summary(client):
    """Return (max_gb, mean_gb, n_workers) for current worker RSS, or None."""
    try:
        import psutil  # noqa: F401
    except ImportError:
        return None
    try:
        rss = client.run(
            lambda: __import__("psutil").Process().memory_info().rss
        )
    except Exception:
        return None
    values = [v for v in rss.values() if isinstance(v, (int, float))]
    if not values:
        return None
    n = len(values)
    return max(values) / 1e9, (sum(values) / n) / 1e9, n


def _n_workers(client):
    """Number of workers currently visible to the client (or -1 if unknown)."""
    try:
        info = client.scheduler_info()
    except Exception:
        return -1
    workers = info.get("workers") if isinstance(info, dict) else None
    return len(workers) if workers is not None else -1


def _pending_tasks(client):
    try:
        processing = client.processing()
    except Exception:
        return None
    try:
        return sum(len(v) for v in processing.values())
    except Exception:
        return None


class ProgressTicker:
    """Context manager: spawn a daemon thread emitting periodic status lines."""

    def __init__(self, tag: str, interval: float = 15.0):
        self.tag = tag
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"mxalign-tick-{self.tag}"
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval)

    def _run(self):
        client = _get_client()
        last_tick = time.perf_counter()
        last_n_workers: int | None = None
        warned_no_workers = False
        while not self._stop.wait(self.interval):
            now = time.perf_counter()
            elapsed = now - self._t0
            delta = now - last_tick
            last_tick = now
            parts = [
                f"[mxalign] tick phase={self.tag}",
                f"elapsed={elapsed:.1f}s",
                f"since_last_tick={delta:.1f}s",
            ]
            n_workers = -1
            if client is not None:
                pending = _pending_tasks(client)
                if pending is not None:
                    parts.append(f"pending={pending}")
                n_workers = _n_workers(client)
                parts.append(f"workers={n_workers}")
                rss = _worker_rss_summary(client)
                if rss is not None:
                    max_gb, mean_gb, _ = rss
                    parts.append(f"rss_max_gb={max_gb:.2f}")
                    parts.append(f"rss_mean_gb={mean_gb:.2f}")
            LOG.info(" ".join(parts))
            if (
                client is not None
                and not warned_no_workers
                and last_n_workers is not None
                and last_n_workers > 0
                and n_workers == 0
            ):
                LOG.warning(
                    "[mxalign] no workers visible to client (was %d); scheduler "
                    "likely lost the worker (e.g. heartbeat timeout). Subsequent "
                    "ticks will report stale state.",
                    last_n_workers,
                )
                warned_no_workers = True
            last_n_workers = n_workers


def log_phase_start(tag: str, **kv) -> None:
    extras = " ".join(f"{k}={v}" for k, v in kv.items())
    LOG.info(f"[mxalign] phase={tag} status=start {extras}".rstrip())


def log_phase_done(tag: str, elapsed: float, **kv) -> None:
    extras = " ".join(f"{k}={v}" for k, v in kv.items())
    LOG.info(
        f"[mxalign] phase={tag} status=done elapsed={elapsed:.2f}s {extras}".rstrip()
    )


def log_dashboard() -> None:
    client = _get_client()
    if client is None:
        return
    link = getattr(client, "dashboard_link", None)
    if link:
        LOG.info(f"[mxalign] dask dashboard={link}")
