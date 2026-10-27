"""Metric computation for the verification step of the pipeline."""

from functools import partial

import xarray as xr

from .transformations.external import _resolve_function


class Metric:
    """A verification metric with the reference dataset bound in.

    Parameters
    ----------
    name : str
        Name of the metric.
    func_path : str
        Dotted import path of the metric function.
    ds_ref : xr.Dataset
        Reference dataset passed to the arguments marked ``"reference"``.
    inputs : dict of str to str
        Mapping of function argument names to the dataset role passed to
        them: ``"reference"`` for ``ds_ref``; any other value marks the
        argument that receives the dataset given to :meth:`compute`.
    **kwargs
        Additional keyword arguments passed on to the metric function.
        ``dim`` is also used to rechunk the datasets for ``xskillscore``
        functions.

    Raises
    ------
    ValueError
        If more than one argument in ``inputs`` is not ``"reference"``.
    ImportError
        If the module in ``func_path`` cannot be imported.
    AttributeError
        If the module has no attribute with the function name.
    IndexError
        If no argument in ``inputs`` is a non-reference (forecast) argument.
    """

    def __init__(
        self,
        name: str,
        func_path: str,
        ds_ref: xr.Dataset,
        inputs: dict[str, str],
        **kwargs: object,
    ) -> None:
        self.name = name
        func = _resolve_function(func_path)
        self._is_xskillscore = func.__module__.startswith("xskillscore")
        self._dim = kwargs.get("dim", None)

        kwarg_ref = {}
        kwarg_ds = []
        for input_arg, ds_type in inputs.items():
            if ds_type == "reference":
                kwarg_ref[input_arg] = (
                    self._rechunk(ds_ref) if self._is_xskillscore else ds_ref
                )
            else:
                kwarg_ds.append(input_arg)
        if len(kwarg_ds) > 1:
            raise ValueError(
                f"More than one predictor-input argument defined for function {func_path}"
            )
        partial_kwargs = {**kwarg_ref, **kwargs}
        self._func = partial(func, **partial_kwargs)
        self._kwarg_ds = kwarg_ds[0]

    def compute(self, ds: xr.Dataset) -> xr.Dataset:
        """Compute the metric of ``ds`` against the reference dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset (e.g. a forecast) to verify.

        Returns
        -------
        xr.Dataset
            The result of the metric function.
        """
        if self._is_xskillscore:
            ds = self._rechunk(ds)
        kwarg_ds = {self._kwarg_ds: ds}
        return self._func(**kwarg_ds)

    def _rechunk(self, ds: xr.Dataset) -> xr.Dataset:
        """Rechunk ``ds`` to one chunk along ``dim`` and size-1 chunks elsewhere.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to rechunk.

        Returns
        -------
        xr.Dataset
            The rechunked dataset, or ``ds`` unchanged if no ``dim`` was given.
        """
        if self._dim is None:
            return ds
        dim = [self._dim] if isinstance(self._dim, str) else self._dim
        dim_other = [d for d in ds.dims if d not in dim]
        chunks = {d: -1 for d in dim}
        for d in dim_other:
            chunks[d] = 1
        return ds.chunk(chunks)


def verify(
    fcst: xr.Dataset,
    obs: xr.Dataset,
    func_path: str,
    inputs: dict[str, str],
    **kwargs: object,
) -> xr.Dataset:
    """Compute a metric function of a forecast against an observation.

    Parameters
    ----------
    fcst : xr.Dataset
        Forecast dataset.
    obs : xr.Dataset
        Observation dataset.
    func_path : str
        Dotted import path of the metric function.
    inputs : dict of str to str
        Mapping of function argument names to ``"forecast"`` or
        ``"observation"``, selecting which dataset each argument receives.
    **kwargs
        Additional keyword arguments passed on to the metric function.

    Returns
    -------
    xr.Dataset
        The result of the metric function.

    Raises
    ------
    KeyError
        If a value in ``inputs`` is neither ``"forecast"`` nor
        ``"observation"``.
    ImportError
        If the module in ``func_path`` cannot be imported.
    AttributeError
        If the module has no attribute with the function name.
    """
    func = _resolve_function(func_path=func_path)
    datasets = {
        "forecast": fcst,
        "observation": obs,
    }
    input_kwargs = {arg_name: datasets[ds_type] for arg_name, ds_type in inputs.items()}

    all_kwargs = {**input_kwargs, **kwargs}

    result = func(**all_kwargs)
    return result
