import numpy as np
from earthkit.data.utils.patterns import Pattern


class Dates:
    def __init__(self, start: str | np.datetime64, end: str | np.datetime64, period: str | np.timedelta64, range: str | np.timedelta64, step: str | np.timedelta64):
        self._start = np.datetime64(start) if not isinstance(start, np.datetime64) else start
        self._end = np.datetime64(end) if not isinstance(end, np.datetime64) else end
        self._period = to_timedelta64(period) if isinstance(period, str) else period
        self._range = to_timedelta64(range) if isinstance(range, str) else range
        self._step = to_timedelta64(step) if isinstance(step, str) else step
        valid_times = set()
        lead_times = set()
        reference_times = set()
        date = self._start
        while date <= self._end:
            print(date)
            reference_times.add(date)
            delta = np.timedelta64(0,"s")
            while delta <= self._range:
                valid_times.add(date + delta)
                lead_times.add(delta)
                delta += self._step
            date += self._period
        self.valid_times = list(valid_times)
        self.reference_times = list(reference_times)
        # FIXME: can we simplify this? earthkit.data.utils.patterns.Pattern does not accept np.int64
        self.lead_times = sorted([int(t.astype(int)) for t in lead_times])

    def substitute(self, path: str):
        pattern = Pattern(path)
        paths = pattern.substitute(
                dict(reference_time=self.reference_times),
                # dict(lead_time=self.lead_times),
                # dict(valid_time=self.valid_times),
                allow_extra=True
            )
        return sorted(paths)
       
        
def to_timedelta64(freq: str) -> np.timedelta64:
    """
    Convert a frequency string to a numpy timedelta64 object.
    The frequency string should be in the format of a number followed by a time unit,
    e.g. '1D', '2H', '3M', etc.
    The time unit can be one of the following:
    - 'Y' for years
    - 'M' for months
    - 'W' for weeks
    - 'D' for days
    - 'h' for hours
    - 'm' for minutes
    - 's' for seconds
    - 'ms' for milliseconds
    Parameters
    ----------
    freq : str
        The frequency string to convert.
    
    Returns
    -------
    np.timedelta64
        The converted numpy timedelta64 object.
    """
    value = freq[:-1]
    unit = freq[-1]
    return np.timedelta64(value,unit)