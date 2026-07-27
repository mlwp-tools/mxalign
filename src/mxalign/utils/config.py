import numpy as np
import yaml

from .dates import Dates


def load_yaml(fn: str) -> dict:
    with open(fn, "r") as f:
        return yaml.safe_load(f)


class Config:
    def __init__(self, config: str | dict):
        self.config = load_yaml(config) if isinstance(config, str) else config
        if not isinstance(self.config, dict):
            raise TypeError("config should be a dictionary.")
        self.dates = self.config.pop("dates", None)
        self._init_datasets()
        print("Config initialized")

    def __getitem__(self, key):
        config = self.config.get(key, None)
        if config:
            return config.copy()
        else:
            return config

    def __call__(self):
        return self.config

    def _init_datasets(self):
        for key, loader in self.config["datasets"].items():
            dates_loader = loader.pop("dates", None)
            if self.dates:
                if dates_loader:
                    keys_all = list(set(self.dates.keys()).union(dates_loader.keys()))
                    dates = {
                        key: (
                            dates_loader[key]
                            if key in dates_loader.keys()
                            else self.dates[key]
                        )
                        for key in keys_all
                    }
                else:
                    dates = self.dates.copy()
            else:
                if dates_loader:
                    dates = dates_loader.copy()
                else:
                    dates = None

            if dates:
                dates = Dates(**dates)
                loader["files"] = dates.substitute(loader["files"])
                # Propagate declarative time hints to every loader.
                # BaseLoader.load() uses these to pre-prune datasets:
                #   - `valid_times` prunes observation datasets (1D dim).
                #   - `reference_times` + `lead_times` prune forecast
                #     datasets rectangularly, enforcing `dates.range`
                #     (max lead) and `dates.period` (rt spacing).
                loader.setdefault(
                    "valid_times",
                    np.sort(np.array(dates.valid_times)),
                )
                loader.setdefault(
                    "reference_times",
                    np.sort(np.array(dates.reference_times)),
                )
                loader.setdefault(
                    "lead_times",
                    # dates.lead_times strips unit info (stored as plain ints).
                    # Reconstruct from _step/_range to keep the timedelta64 unit
                    # so that BaseLoader can cast correctly to the dataset dtype.
                    np.arange(int(dates._range / dates._step) + 1) * dates._step,
                )
            self.config["datasets"][key] = loader
