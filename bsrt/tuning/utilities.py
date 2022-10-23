from typing import Dict, List, TypeVar, cast

_T = TypeVar("_T")


def prepend_to_param_names(
    prefix: str, params: List[Dict[str, _T]]
) -> List[Dict[str, _T]]:
    return [dict(param, name=cast(_T, f"{prefix}.{param['name']}")) for param in params]


def filter_and_remove_from_keys(prefix: str, params: Dict[str, _T]) -> Dict[str, _T]:
    return {
        k.replace(f"{prefix}.", "", 1): v
        for k, v in params.items()
        if k.startswith(prefix)
    }
