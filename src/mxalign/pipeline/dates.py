"""Expansion of date-templated file path patterns over reference and lead times."""

import itertools
import re
import string
from datetime import datetime

import numpy as np
import pandas as pd

HOUR = pd.Timedelta("1h")


class Dates:
    """Enumerate reference and lead times over a regular date range.

    Parameters
    ----------
    start : str or datetime or np.datetime64
        First reference time of the range.
    end : str or datetime or np.datetime64
        Last reference time of the range.
    period : str or np.timedelta64
        Step between successive reference times (e.g. ``"1D"``).
    range : str or np.timedelta64
        Forecast horizon — lead times run from 0 to ``range`` inclusive.
    step : str or np.timedelta64
        Increment between lead times (e.g. ``"1h"``).
    """

    def __init__(
        self,
        start: str | datetime | np.datetime64,
        end: str | datetime | np.datetime64,
        period: str | np.timedelta64,
        range: str | np.timedelta64,
        step: str | np.timedelta64,
    ) -> None:
        self.reference_times = pd.date_range(start, end, freq=pd.Timedelta(period))
        self.lead_times = pd.timedelta_range(
            pd.Timedelta(0), pd.Timedelta(range), freq=pd.Timedelta(step)
        )

    def substitute(self, path: str) -> list[str]:
        """Expand ``path`` over all (reference_time, lead_time) pairs.

        Parameters
        ----------
        path : str
            Path pattern in :meth:`str.format` syntax with the fields
            ``reference_time`` and ``valid_time`` (datetimes, e.g.
            ``{reference_time:%Y%m%d%H}``) and ``lead_time`` (integer hours,
            e.g. ``{lead_time:03d}``).

        Returns
        -------
        list of str
            Sorted unique expanded paths.

        Raises
        ------
        ValueError
            If ``path`` uses ``{lead_time}`` and a lead time is not a whole
            number of hours.
        """
        fields = _field_names(path)
        if "lead_time" in fields and (self.lead_times % HOUR != pd.Timedelta(0)).any():
            raise ValueError(
                "{lead_time} is formatted in whole hours, so step must be a multiple of 1h."
            )
        # Only loop over lead times when the path depends on them
        lead_times = (
            self.lead_times
            if fields & {"lead_time", "valid_time"}
            else self.lead_times[:1]
        )
        paths = {
            path.format(
                reference_time=ref, lead_time=lead // HOUR, valid_time=ref + lead
            )
            for ref, lead in itertools.product(self.reference_times, lead_times)
        }
        return sorted(paths)


def _field_names(path: str) -> set[str]:
    """Return the top-level replacement field names used in a format pattern.

    Parameters
    ----------
    path : str
        Pattern in :meth:`str.format` syntax.

    Returns
    -------
    set of str
        Field names, with attribute access and indexing stripped (e.g.
        ``{a.b}`` and ``{a[0]}`` both give ``"a"``).
    """
    return {
        re.split(r"[.\[]", name)[0]
        for _, name, _, _ in string.Formatter().parse(path)
        if name
    }
