_TRANSFORMATION_REGISTRY = {}
_SIGNATURE_REGISTRY = {}
_EXPANDER_REGISTRY = {}


def register_transformation(name, signature=None):
    """Register a transformation function under ``name``.

    Parameters
    ----------
    name
        Registry key, also the value used in YAML ``transformations:`` blocks.
    signature
        Optional callable ``(**kwargs) -> (inputs, outputs)`` returning the
        lists of source and sink variable names for the transformation given
        its YAML kwargs. Used by consumers (e.g. the fused verification
        engine) to determine, without executing, which variables a
        transformation reads from and writes to a dataset. Transformations
        whose I/O cannot be derived from kwargs alone may omit it; callers
        that need the information must then either fall back or fail.
    """

    def decorator(func):
        _TRANSFORMATION_REGISTRY[name] = func
        if signature is not None:
            _SIGNATURE_REGISTRY[name] = signature
        return func

    return decorator


def available_transformations():
    return list(_TRANSFORMATION_REGISTRY.keys())


def get_transformation(name):
    try:
        return _TRANSFORMATION_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown transformation: {name}")


def get_signature(name):
    """Return the variable I/O signature callable for ``name``, or ``None``.

    The callable, when invoked with the transformation's YAML kwargs,
    returns ``(inputs, outputs)`` — two lists of variable names. ``None``
    means the transformation did not declare a signature.
    """
    return _SIGNATURE_REGISTRY.get(name)


def register_expander(name):
    """Register a pre-execution expander for transformation ``name``.

    An expander is a callable ``(ds, kwargs: dict) -> dict`` that receives
    the current dataset and the raw YAML kwargs, and returns a new kwargs
    dict with glob patterns expanded and optional defaults filled in.  The
    runner calls the expander (if present) before both executing the
    transformation and recording its kwargs in ``_transforms_by_ds``, so
    that downstream engines always see concrete, fully-resolved variable
    names.
    """
    def decorator(func):
        _EXPANDER_REGISTRY[name] = func
        return func
    return decorator


def get_expander(name):
    """Return the expander callable for ``name``, or ``None``."""
    return _EXPANDER_REGISTRY.get(name)
