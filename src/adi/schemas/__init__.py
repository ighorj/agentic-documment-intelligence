"""Domain schemas. Register new verticals here to make them available by name."""

from adi.schemas.base import DomainSchema, FieldSpec
from adi.schemas.finance import SCHEMA as FINANCE
from adi.schemas.healthcare import SCHEMA as HEALTHCARE

GENERIC = DomainSchema(name="generic", fields=[])

_REGISTRY: dict[str, DomainSchema] = {
    "generic": GENERIC,
    "healthcare": HEALTHCARE,
    "finance": FINANCE,
}


def get_schema(name: str) -> DomainSchema:
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise KeyError(f"unknown domain '{name}'. Available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[key]


def available_domains() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["DomainSchema", "FieldSpec", "get_schema", "available_domains"]
