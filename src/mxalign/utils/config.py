import yaml

from .dates import Dates


def load_yaml(fn: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(fn, "r") as f:
        return yaml.safe_load(f)


class Config:
    """Pipeline configuration loaded from a YAML file or a plain dict.

    Top-level ``dates`` entries are merged into each dataset's loader config
    so that file-path patterns can be expanded to concrete paths.
    """

    def __init__(self, config: str | dict) -> None:
        self.config = load_yaml(config) if isinstance(config, str) else config
        if not isinstance(self.config, dict):
            raise TypeError("config should be a dictionary.")
        self.dates = self.config.pop("dates", None)
        self._init_datasets()
        print("Config initialized")

    def __getitem__(self, key: str) -> dict | None:
        config = self.config.get(key, None)
        if config:
            return config.copy()
        else:
            return config

    def __call__(self) -> dict:
        return self.config

    def _init_datasets(self) -> None:
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
            self.config["datasets"][key] = loader
