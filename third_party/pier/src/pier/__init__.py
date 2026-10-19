import importlib.metadata

from pier.constants import PYPI_PACKAGE_NAME

__version__ = importlib.metadata.version(PYPI_PACKAGE_NAME)
