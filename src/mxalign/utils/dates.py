import numpy as np
from earthkit.data.utils.patterns import Pattern


class Dates:
    """Enumerate reference/valid/lead times over a regular date range.

    Parameters
    ----------
    start, end:
        First and last reference time of the range.
    period:
        Step between successive reference times (e.g. ``"1D"``).
    range:
        Forecast horizon — lead times run from 0 to ``range`` inclusive.
    step:
        Increment between lead times (e.g. ``"1h"``).
    """

    def __init__(
        self,
        start: str | np.datetime64,
        end: str | np.datetime64,
        period: str | np.timedelta64,
        range: str | np.timedelta64,
        step: str | np.timedelta64,
    ):
        self._start = (
            np.datetime64(start) if not isinstance(start, np.datetime64) else start
        )
        self._end = np.datetime64(end) if not isinstance(end, np.datetime64) else end
        self._period = to_timedelta64(period) if isinstance(period, str) else period
        self._range = to_timedelta64(range) if isinstance(range, str) else range
        self._step = to_timedelta64(step) if isinstance(step, str) else step
        valid_times = set()
        lead_times = set()
        reference_times = set()
        date = self._start
        while date <= self._end:
            # print(date)
            reference_times.add(date)
            delta = np.timedelta64(0, "s")
            while delta <= self._range:
                valid_times.add(date + delta)
                lead_times.add(delta)
                delta += self._step
            date += self._period
        self.valid_times = list(valid_times)
        self.reference_times = list(reference_times)
        # FIXME: can we simplify this? earthkit.data.utils.patterns.Pattern does not accept np.int64
        self.lead_times = sorted([int(t.astype(int)) for t in lead_times])

    def substitute(self, path: str) -> list[str]:
        """Expand ``path`` pattern over all reference times; return sorted file list."""
        pattern = Pattern(path)
        paths = pattern.substitute(
            (dict(reference_time=self.reference_times),),
            # dict(lead_time=self.lead_times),
            # dict(valid_time=self.valid_times),
            allow_extra=True,
        )
        return sorted(paths)


def to_timedelta64(freq: str) -> np.timedelta64:
    """Convert a frequency string such as ``"6h"`` or ``"1D"`` to :class:`numpy.timedelta64`."""
    value = freq[:-1]
    unit = freq[-1]
    return np.timedelta64(value, unit)
