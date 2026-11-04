"""A tiny dependency-free schema/validator for LLM JSON output.

Deliberately small: enough to declare required fields, types, enums, defaults,
and list-element coercion, without pulling in pydantic. If validation needs
grow, ``Schema.validate`` is the single seam to swap in a heavier library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence


class ValidationError(Exception):
    """Raised only when Schema.validate is called with strict=True."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


_PY_TYPES = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}


@dataclass
class Field:
    """A single expected field in an LLM JSON object.

    Parameters
    ----------
    type:
        One of ``str|int|float|bool|list|dict``.
    required:
        If True and the key is missing (or None), an error is recorded and the
        ``default`` is used.
    default:
        Value substituted when the key is absent/invalid.
    choices:
        Optional allowed values (compared case-insensitively for strings).
    item_type:
        For ``list`` fields, coerce each element to this scalar type; elements
        that fail are dropped (and counted).
    coerce:
        If True, attempt best-effort type coercion (e.g. "3" -> 3) before
        recording an error.
    """

    type: str
    required: bool = False
    default: Any = None
    choices: Optional[Sequence[Any]] = None
    item_type: Optional[str] = None
    coerce: bool = True

    def __post_init__(self) -> None:
        if self.type not in _PY_TYPES:
            raise ValueError(f"Unsupported field type: {self.type!r}")
        if self.item_type is not None and self.item_type not in _PY_TYPES:
            raise ValueError(f"Unsupported item_type: {self.item_type!r}")


@dataclass
class Schema:
    """A named set of :class:`Field` definitions."""

    name: str
    fields: dict[str, Field] = field(default_factory=dict)

    def validate(
        self,
        data: Any,
        *,
        strict: bool = False,
    ) -> tuple[dict[str, Any], list[str]]:
        """Validate/coerce ``data`` against this schema.

        Returns ``(clean_dict, errors)``. Unknown keys are passed through
        unchanged so downstream code can still read extra model output. When
        ``strict`` is True and any error occurs, raises :class:`ValidationError`.
        """
        errors: list[str] = []
        if not isinstance(data, dict):
            errors.append(f"{self.name}: expected object, got {type(data).__name__}")
            clean = {k: f.default for k, f in self.fields.items()}
            if strict:
                raise ValidationError(errors)
            return clean, errors

        clean: dict[str, Any] = dict(data)  # keep unknown keys
        for key, spec in self.fields.items():
            present = key in data and data[key] is not None
            if not present:
                if spec.required:
                    errors.append(f"{self.name}.{key}: required field missing")
                clean[key] = spec.default
                continue
            value, err = self._coerce_value(key, spec, data[key])
            if err:
                errors.append(err)
                clean[key] = spec.default
            else:
                clean[key] = value

        if strict and errors:
            raise ValidationError(errors)
        return clean, errors

    # ----------------------------------------------------------------- #
    def _coerce_value(self, key: str, spec: Field, raw: Any) -> tuple[Any, Optional[str]]:
        target = _PY_TYPES[spec.type]
        value = raw

        if spec.type == "list":
            if not isinstance(raw, list):
                return None, f"{self.name}.{key}: expected list, got {type(raw).__name__}"
            if spec.item_type is not None:
                value = self._coerce_list_items(spec, raw)
            return value, None

        if not isinstance(raw, target):
            if spec.coerce:
                coerced = _try_coerce_scalar(target, raw)
                if coerced is None:
                    return None, (
                        f"{self.name}.{key}: expected {spec.type}, "
                        f"got {type(raw).__name__}"
                    )
                value = coerced
            else:
                return None, (
                    f"{self.name}.{key}: expected {spec.type}, got {type(raw).__name__}"
                )

        if spec.choices is not None:
            if not _in_choices(value, spec.choices):
                return None, (
                    f"{self.name}.{key}: {value!r} not in {list(spec.choices)}"
                )
        return value, None

    def _coerce_list_items(self, spec: Field, raw: list) -> list:
        item_target = _PY_TYPES[spec.item_type]  # type: ignore[index]
        out = []
        for item in raw:
            if isinstance(item, item_target):
                out.append(item)
            else:
                coerced = _try_coerce_scalar(item_target, item)
                if coerced is not None:
                    out.append(coerced)
                # silently dropped items are reflected by length; callers that
                # care can compare len(out) vs len(raw).
        return out


def _try_coerce_scalar(target: type, raw: Any) -> Any:
    try:
        if target is bool:
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "yes", "on"}
            return bool(raw)
        if target in (int, float, str):
            return target(raw)
    except (ValueError, TypeError):
        return None
    return None


def _in_choices(value: Any, choices: Sequence[Any]) -> bool:
    if isinstance(value, str):
        low = value.strip().lower()
        return any(isinstance(c, str) and c.strip().lower() == low for c in choices)
    return value in choices
