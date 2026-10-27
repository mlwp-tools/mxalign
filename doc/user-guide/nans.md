# Consistent missing data

Why datasets should share one NaN mask before scoring, and how
`mxalign.align.nans.broadcast_nans` builds that mask across a list or dict
of aligned datasets: which coordinates and variables it compares, and its
cost on large inputs. It currently works on in-memory data only; see the
warning in [How alignment works](../concepts/alignment.md#missing-data).

!!! note "Coming soon"
    This page is still being written. For now, see the
    [API reference](../reference/align.md#mxalign.align.nans.broadcast_nans).
