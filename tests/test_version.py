import mxalign


def test_version_is_set() -> None:
    """Check that the package version is resolved from the git tag.

    A fallback version of ``0.0.0`` means setuptools-scm could not find any
    git metadata when the package was installed.
    """
    assert isinstance(mxalign.__version__, str)
    assert mxalign.__version__ not in ("", "0.0.0")
