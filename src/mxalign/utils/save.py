import xarray as xr
from earthkit.data.utils.patterns import Pattern


class DatasetPath:
    """Resolve a path pattern to a concrete file path using date components from a dataset.

    The dominant year/month/day (by count) in ``reference_time`` (forecast) or
    ``valid_time`` (observation) is used to fill ``{year}``, ``{month}``, ``{day}``
    placeholders via :meth:`substitute`.
    """

    def __init__(self, name: str, ds: xr.Dataset) -> None:
        self.name = name
        if ds.mx.is_forecast():
            years = ds["reference_time"].groupby(ds["reference_time"].dt.year).count()
            self.year = int(years.isel(year=years.argmax())["year"].values)
            ds_month = ds.sel(reference_time=ds.reference_time.dt.year == self.year)
            months = (
                ds_month["reference_time"]
                .groupby(ds_month["reference_time"].dt.month)
                .count()
            )
            self.month = int(months.isel(month=months.argmax())["month"].values)
            ds_day = ds_month.sel(
                reference_time=ds_month.reference_time.dt.month == self.month
            )
            days = (
                ds_day["reference_time"]
                .groupby(ds_day["reference_time"].dt.day)
                .count()
            )
            self.day = int(days.isel(day=days.argmax())["day"].values)
        elif ds.mx.is_observation():
            years = ds["valid_time"].groupby(ds["valid_time"].dt.year).count()
            self.year = int(years.isel(year=years.argmax())["year"].values)
            ds_month = ds.sel(valid_time=ds.valid_time.dt.year == self.year)
            months = (
                ds_month["valid_time"].groupby(ds_month["valid_time"].dt.month).count()
            )
            self.month = int(months.isel(month=months.argmax())["month"].values)
            ds_day = ds_month.sel(valid_time=ds_month.valid_time.dt.month == self.month)
            days = ds_day["valid_time"].groupby(ds_day["valid_time"].dt.day).count()
            self.day = int(days.isel(day=days.argmax())["day"].values)

    def substitute(self, path: str) -> str:
        """Expand ``path`` pattern with ``{name}``, ``{year}``, ``{month}``, ``{day}``."""
        pattern = Pattern(path)
        path = pattern.substitute(
            dict(name=self.name),
            dict(year=self.year),
            dict(month=self.month),
            dict(day=self.day),
            allow_extra=True,
        )
        return path


def save_dataset(method: str, name: str, ds: xr.Dataset, **kwargs) -> None:
    """Save ``ds`` to a path derived from its dominant date using ``ds.<method>``."""
    save_fn = getattr(ds, method)
    dataset = DatasetPath(name, ds)
    path = dataset.substitute(kwargs.pop("path"))
    print(f"Saving to {path}")
    save_fn(path, **kwargs)


def save_metrics(method: str, ds: xr.Dataset, **kwargs) -> None:
    """Save ``ds`` to a fixed path using ``ds.<method>``."""
    save_fn = getattr(ds, method)
    path = kwargs.pop("path")
    print(f"Saving to {path}")
    save_fn(path, **kwargs)
