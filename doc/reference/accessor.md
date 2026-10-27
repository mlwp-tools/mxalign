# `ds.mx` accessor

`import mxalign` registers this class as the `mx` accessor on every
`xr.Dataset`. You never create it yourself: call its methods as
`ds.mx.<method>(...)`. None of the methods modify `ds`; operations return
new datasets.

::: mxalign.accessors.mx.MxAccessor
    options:
      show_root_heading: false
      members_order: source
