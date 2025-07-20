import importlib
import pkgutil
import inspect
import sys

__all__ = []

# Get the __path__ attribute of the current package
package_path = sys.modules[__name__].__path__

computer_classes = []

modules_added = set()

# Dynamically import all classes from all modules in this package
for loader, module_name, is_pkg in pkgutil.iter_modules(package_path):
    module = importlib.import_module(f"{__name__}.{module_name}")
    if module not in modules_added:
        __all__.append(module) # type: ignore
        modules_added.add(module)
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Only import classes defined in the module (not imported ones)
        if obj.__module__ == module.__name__:
            globals()[name] = obj
            computer_classes.append(obj)
