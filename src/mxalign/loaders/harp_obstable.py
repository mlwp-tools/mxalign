import numpy as np
import xarray as xr
import sqlite3
import pandas as pd

from .registry import register_loader
from ..properties.properties import Properties, Space, Time, Uncertainty
from .base import BaseLoader

COORDS = {
    "longitude": "lon",
    "latitude":"lat",
    "valid_time": "validdate",
    "code": "SID",
    "altitude": "elev",
}

@register_loader
class ObstableLoader(BaseLoader):

    name = "harp-obstable"

    space = Space.POINT
    time = Time.OBSERVATION
    uncertainty = Uncertainty.DETERMINISTIC


    def _load(self):
        files = [self.files] if isinstance(self.files, str) else self.files
        if len(files) > 1:
            raise NotImplementedError("Reading from mutliple SQLite-files not implemented")

        conn = sqlite3.connect(files[0])

        if self.variables is None:
            # Retrieve all variables
            variables = [
                var for var in pd.read_sql_query(
                    "SELECT * FROM SYNOP LIMIT 0",
                    conn
                ).columns if var not in COORDS.values() 
            ]
            print(variables)
        else:
            variables = self.variables
        
        # Read the SIDs
        codes = pd.read_sql(
            f"SELECT SID as code, MIN(lat) AS latitude, MIN(lon) AS longitude, elev as altitude FROM SYNOP GROUP BY SID",
            conn,
            index_col="code"
        ).to_xarray()

        print(codes)
        # Read the data
        query = f"""
                SELECT SID as code, validdate as valid_time, {", ".join(variables)}
                FROM SYNOP
            """
        print(query)
        df = pd.read_sql(
                query,
                conn,
                index_col=["code","valid_time"],
                parse_dates={"valid_time": {"unit": "s"}}
            )
        print(df)
        
        ds = df.to_xarray()
        lon_values = codes["longitude"].sel(code=ds["code"]).values
        lat_values = codes["latitude"].sel(code=ds["code"]).values
        alt_values = codes["altitude"].sel(code=ds["code"]).values
        
        ds = ds.assign_coords(
            longitude=("code", lon_values),
            latitude=("code", lat_values),
            altitude=("code", alt_values)
        )
        
        return ds.rename_dims({"code":"point_index"}).transpose("valid_time","point_index")


            