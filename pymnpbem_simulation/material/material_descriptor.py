from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

# need this for the GUI tie-in (importing user materials from files at runtime)
def resolve_refractive_index_paths(ri_paths: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert serializable descriptor specs into runtime objects.
    Supported:
      - {"type": "table", "file": "/abs/path/file.dat"} -> "/abs/path/file.dat"
      - {"type": "python_module", "module_path": "...py", "factory": "generate_eps_func"} -> callable
    """
    if not isinstance(ri_paths, dict):
        return ri_paths

    resolved: Dict[str, Any] = {}

    for name, spec in ri_paths.items():
        if not isinstance(spec, dict):
            resolved[name] = spec
            continue

        stype = str(spec.get("type", "")).lower()

        if stype == "table":
            file_path = spec.get("file")
            if not file_path:
                raise ValueError(f"Material '{name}' table spec missing 'file'")
            resolved[name] = str(file_path)
            continue

        if stype == "python_module":
            module_path = spec.get("module_path")
            factory_name = str(spec.get("factory", "generate_eps_func"))

            if not module_path:
                raise ValueError(f"Material '{name}' python_module spec missing 'module_path'")

            p = Path(str(module_path))
            if not p.exists():
                raise FileNotFoundError(f"Material module not found for '{name}': {module_path}")

            mod_name = f"_mnpbem_user_mat_{p.stem}_{abs(hash(str(p.resolve())))}"
            spec_obj = importlib.util.spec_from_file_location(mod_name, str(p))
            if spec_obj is None or spec_obj.loader is None:
                raise RuntimeError(f"Failed to import material module: {module_path}")

            mod = importlib.util.module_from_spec(spec_obj)
            sys.modules[mod_name] = mod
            spec_obj.loader.exec_module(mod)

            if not hasattr(mod, factory_name):
                raise AttributeError(f"{module_path} missing factory '{factory_name}'")

            fn = getattr(mod, factory_name)()
            if not callable(fn):
                raise TypeError(f"{module_path}:{factory_name} did not return callable")
            resolved[name] = fn
            continue

        # unknown dict spec: pass through
        resolved[name] = spec

    return resolved