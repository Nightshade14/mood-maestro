"""scripts package initializer.

Making `scripts` a package allows project-level imports such as
`import scripts.models` to work when the project root is on sys.path
or when running modules with `python -m` or after installing the package.
"""

__all__ = ["models", "set_db"]
