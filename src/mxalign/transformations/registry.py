_TRANSFORMATION_REGISTRY = {}
_SIGNATURE_REGISTRY = {}


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
