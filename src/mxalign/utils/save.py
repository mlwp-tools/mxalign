from earthkit.data.utils.patterns import Pattern

class DatasetPath():
    def __init__(self, name, ds):
        self.name = name
        if ds.time.is_forecast():
            years = ds["reference_time"].groupby(ds["reference_time"].dt.year).count()
            self.year = int(years.isel(year=years.argmax())["year"].values)
            ds_month = ds.sel(reference_time=ds.reference_time.dt.year == self.year)
            months = ds_month["reference_time"].groupby(ds_month["reference_time"].dt.month).count()
            self.month = int(months.isel(month=months.argmax())["month"].values)
            ds_day = ds_month.sel(reference_time=ds_month.reference_time.dt.month==self.month)
            days = ds_day["reference_time"].groupby(ds_day["reference_time"].dt.day).count()
            self.day = int(days.isel(day=days.argmax())["day"].values)
        elif ds.time.is_observation():
            years = ds["valid_time"].groupby(ds["valid_time"].dt.year).count()
            self.year = int(years.isel(year=years.argmax())["year"].values)
            ds_month = ds.sel(valid_time=ds.valid_time.dt.year == self.year)
            months = ds_month["valid_time"].groupby(ds_month["valid_time"].dt.month).count()
            self.month = int(months.isel(month=months.argmax())["month"].values)
            ds_day = ds_month.sel(valid_time=ds_month.valid_time.dt.month==self.month)
            days = ds_day["valid_time"].groupby(ds_day["valid_time"].dt.day).count()
            self.day = int(days.isel(day=days.argmax())["day"].values)



            
    def substitute(self, path: str):
        pattern = Pattern(path)
        path = pattern.substitute(
                dict(name=self.name),
                dict(year=self.year),
                dict(month=self.month),
                dict(day=self.day),
                allow_extra=True
            )
        return path
    
def save_dataset(method, name, ds, **kwargs):
    ds = ds.drop_attrs(deep=True)
    save_fn = getattr(ds, method)
    dataset = DatasetPath(name, ds)
    path = dataset.substitute(kwargs.pop("path"))
    save_fn(path, **kwargs)

def save_metrics(method, ds, **kwargs):
    ds = ds.drop_attrs(deep=True)
    save_fn = getattr(ds, method)
    path = kwargs.pop("path")
    save_fn(path, **kwargs)
