---
status: proposed
date: 2026-10-27
---

# NaN masks for ensembles: reduce over members, never match them

## Context and Problem Statement

`broadcast_nans` propagates missing values across the aligned datasets, so that every model and the reference are verified on the same sample. It matched every dimension by coordinate value. Ensembles break that assumption. Observations and deterministic forecasts have no `member` dimension, and two ensembles have unrelated member labels and often different sizes (e.g. 5 and 31 members). As a result `broadcast_nans` crashed on any run with an ensemble, and the pipeline runs it by default.

What does "missing" mean for an ensemble value, and how are members compared between datasets?

## Decision Drivers

* Every model and the reference should be verified on the same (time, space) sample.
* Ensemble scores assume a fixed ensemble size M across the sample: fair CRPS and the Brier score are normalised by M, and the spread-skill ratio is compared to `sqrt((M+1)/M)`.
* Member labels identify members within one ensemble only. Member 3 of model A has nothing to do with member 3 of model B.
* mxalign should not depend on the position of the `member` dimension, because its order isn't enforced by mlwp-data-specs.

## Considered Options

1. An ensemble value is missing if **any** member is missing. The mask is computed with `member` reduced away and then applied to all members. Members are never matched between datasets.
2. Keep per-member NaNs, mask the other datasets only where **all** members are missing, and let every metric cope with a varying M.
3. Match `member` by label between ensembles, like any other dimension, and reduce it only against datasets that don't have it.

## Decision Outcome

Chosen option: 1, because it is the only one that keeps M constant over the verification sample without every metric having to handle NaNs itself. It also handles any number of ensembles of any size the same way.

The uncertainty dimension is taken from the uncertainty trait (`member` for `ensemble`, `quantile` for `quantile`), not from dimension names. The same rule applies to quantile datasets. Apart from the uncertainty dimension, a variable shared by two datasets must have the same dimensions in both. Otherwise `broadcast_nans` raises, because the datasets aren't aligned.

### Consequences

* Good, because M is the same at every verified point, and all models and the reference share one sample.
* Good, because a dataset tagged `deterministic` that still has a `member` dimension raises an error instead of being silently matched member by member.
* Bad, because one missing member discards the whole ensemble at that point, e.g. when a single member file is missing. To keep a partial ensemble, subset its members before alignment, or switch `broadcast_nans` off and handle NaNs in the metrics.
* Neutral, because `broadcast_nans` now requires the uncertainty trait on every dataset, which mlwp-data-loaders always sets.

### Confirmation

`tests/test_nans.py` covers observation/ensemble pairs in both orders, observation, deterministic and ensemble together, ensembles of different sizes and labels, dimension order and dask inputs. `tests/test_runner.py::test_align_mixes_deterministic_and_ensemble` covers the pipeline.

## Pros and Cons of the Options

### 2. Per-member NaNs

* Good, because no member data is discarded.
* Bad, because M varies over the sample, so fair scores and the spread-skill reference need per-point M, and samples with different M are averaged together.
* Bad, because every ensemble metric has to mask NaNs jointly itself.

### 3. Match members by label

* Bad, because members of different ensembles are unrelated, so matching labels is meaningless.
* Bad, because ensembles of different sizes would be masked on the overlapping labels only, which again makes M vary.
