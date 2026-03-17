from .transformations.external import _resolve_function
from functools import partial

class Metric():
    def __init__(self, name, func_path, ds_ref, inputs, **kwargs):
        self.name = name
        func = _resolve_function(func_path)
        self._is_xskillscore = func.__module__.startswith("xskillscore")
        self._dim = kwargs.get("dim", None)

        kwarg_ref = {}
        kwarg_ds = []
        for input_arg, ds_type in inputs.items():
            if ds_type == "reference":
                kwarg_ref[input_arg] = self._rechunk(ds_ref) if self._is_xskillscore else ds_ref
            else:
                kwarg_ds.append(input_arg)
        if len(kwarg_ds) > 1:
            raise ValueError(f"More than one predictor-input argument defined for function {func_path}")
        partial_kwargs = {**kwarg_ref, **kwargs}
        self._func = partial(func, **partial_kwargs)
        self._kwarg_ds = kwarg_ds[0]

    def compute(self, ds):
        if self._is_xskillscore:
            ds = self._rechunk(ds)
        kwarg_ds = {self._kwarg_ds: ds}
        return self._func(**kwarg_ds)

    
    def _rechunk(self, ds):
        if self._dim is None:
            return ds
        dim = [self._dim] if isinstance(self._dim, str) else self._dim
        if 'member' in ds.dims:
            dim.append('member') 
        dim_other = [d for d in ds.dims if d not in dim]
        chunks = {d: -1 for d in dim}
        for d in dim_other:
            chunks[d] = 1
        return ds.chunk(chunks)
            

def verify(fcst, obs, func_path, inputs, **kwargs):
    func = _resolve_function(func_path=func_path)
    datasets = {
        "forecast": fcst,
        "observation": obs,
    }
    input_kwargs = {
        arg_name: datasets[ds_type]
        for arg_name, ds_type in inputs.items() 
    }

    all_kwargs = {**input_kwargs, **kwargs}

    result = func(**all_kwargs)
    return(result)