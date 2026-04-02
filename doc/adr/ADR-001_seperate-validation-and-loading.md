---
# These are optional metadata elements. Feel free to remove any of them.
status: accepted
date: 2026-03-25
decision-makers: Denby L. Van Ginderachter M.
informed: Francois B., Buurman S.
---

# Seperation of concerns between dataset-loading, -validation and -alignment

## Context

Currently `mxalign` implements dataset loading functionality, validation functionality (checking if the loaded dataset has all the correct metadata and) and alignment functionality. However, dataset-loading and -validation fall outside the main scope of the `mxalign` package. 

<!-- This is an optional element. Feel free to remove. -->
## Decision Drivers

* Provide clarity on what the package does by seperating the different concerns
* Provide a clear interface for users who want to bring their own dataset
* Provide flexibility for users
* Ease of maintainance by decoupling concerns

## Considered Options

1. Split out both dataset validation `mlwp-data-specs` and dataset loading `mlwp-data-loaders`
2. Split out dataset-validation and loading but keep those two together

## Decision Outcome

Chosen option 1., because it allows for most flexibility and provides a clear entry point for users who want to bring their own loader. 

<!-- This is an optional element. Feel free to remove. -->
### Consequences
* Users can now validate their dataset with CLI
* Simplification of the loader structure; only functions no classes
* `mxalign` is now only responsible for the alignment tasks

## More Information
Currently the interface between dataset loaded with an `mlwp-data-loaders` loader and `mxalign` is not defined. Ideally `mxalign` should know the traits of the dataset to correctly align dataset. How do we inform `mxalign` on the traits? See [ADR-002](./ADR-002_mxalign-loader-interface.md) for possible options. 
