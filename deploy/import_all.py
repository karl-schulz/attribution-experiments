
import pkgutil
import inspect

# IMPORT AAAAAAAAALL
def import_all(path):
    all = []
    for loader, name, is_pkg in pkgutil.walk_packages(path):
        module = loader.find_module(name).load_module(name)
        for name, value in inspect.getmembers(module):
            if name.startswith('__'):
                continue
            globals()[name] = value
            all.append(name)
    return all
