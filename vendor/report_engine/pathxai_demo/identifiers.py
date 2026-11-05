"""Opt-in replacement of declared case/slide identifiers at the VLM boundary.

This is pseudonymization of known text identifiers, not general de-identification:
free-text identifiers, image pixels and other output files still need review.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid


def _replace(value, mapping):
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return value
        # One pass prevents aliases/replacements from being substituted again.
        pattern = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
        return pattern.sub(lambda match: mapping[match.group()], value)
    if isinstance(value, list):
        return [_replace(v, mapping) for v in value]
    if isinstance(value, tuple):
        return tuple(_replace(v, mapping) for v in value)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            new_key = _replace(key, mapping) if isinstance(key, str) else key
            if new_key in out:
                raise ValueError("Identifier replacement produced duplicate keys")
            out[new_key] = _replace(item, mapping)
        return out
    return value


class IdentifierClient:
    """Masks before recording/sending, restores references for the local pipeline."""
    def __init__(self, inner, mapping):
        if not mapping or any(not isinstance(k, str) or not k or not isinstance(v, str)
                              or not re.fullmatch(r"PXID_[0-9a-f]{16}_[0-9]+", v)
                              for k, v in mapping.items()):
            raise ValueError("Invalid identifier mapping")
        if len(set(mapping.values())) != len(mapping) or any(k in v for k in mapping for v in mapping.values()):
            raise ValueError("Identifier aliases must be unique and independent of originals")
        self.inner, self.mapping = inner, dict(mapping)
        self.reverse = {v: k for k, v in mapping.items()}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def chat(self, prompt, image_paths=None, system_prompt=None, messages=None, **kwargs):
        return self._call("chat", prompt, image_paths, system_prompt, messages, kwargs)

    def chat_json(self, prompt, image_paths=None, system_prompt=None, messages=None, **kwargs):
        return self._call("chat_json", prompt, image_paths, system_prompt, messages, kwargs)

    def _call(self, method, prompt, images, system, messages, kwargs):
        # Paths stay local for encoding; the recorder stores image hashes.
        inputs = {"prompt": prompt, "system_prompt": system, "messages": messages, "kwargs": kwargs}
        serialized = json.dumps(inputs, ensure_ascii=False)
        if any(alias in serialized for alias in self.reverse):
            raise ValueError("Input unexpectedly contains a reserved identifier alias")
        masked = _replace(inputs, self.mapping)
        response = getattr(self.inner, method)(
            masked["prompt"], image_paths=images, system_prompt=masked["system_prompt"],
            messages=masked["messages"], **masked["kwargs"])
        return _replace(response, self.reverse)


def configure_identifiers(client, *, config, case, candidates, output_dir):
    mode = str(config.get("identifier_mode") or "preserve")
    if mode == "preserve":
        if config.get("identifier_map_path"):
            raise ValueError("identifier_map_path requires pseudonymize mode")
        return client, {"mode": "preserve", "known_identifiers_replaced": False}
    if mode != "pseudonymize" or str(config.get("trace_mode") or "record") not in {"record", "replay"}:
        raise ValueError("Use identifier_mode preserve/pseudonymize; pseudonymize requires record or replay")
    trace_mode = str(config.get("trace_mode") or "record")
    source = str(config.get("identifier_map_path") or "")
    known = {str(case.get("case_id") or case.get("id") or "")}
    paths = list(case.get("slide_paths") or [])
    for candidate in candidates:
        for source_fields in (candidate, candidate.get("metadata") or {}):
            for key in ("slide_id", "slide_path", "wsi_path"):
                if source_fields.get(key):
                    paths.append(str(source_fields[key]))
    for item in paths:
        known.update((str(item), Path(item).name, Path(item).stem))
    known.discard("")
    if trace_mode == "replay" and not source:
        raise ValueError("Pseudonymized replay requires the original private identifier_map_path")
    if source:
        data = json.loads(Path(source).read_text())
        if data.get("schema") != "pathxai-identifier-map-v1":
            raise ValueError("Unsupported identifier map")
        mapping = data["mapping"]
    else:
        # Short IDs can also be clinical numbers/words; require explicit review.
        if any(len(value) < 4 or value.isdigit() for value in known):
            raise ValueError("Short/numeric identifiers require an explicit reviewed identifier map")
        prefix = "PXID_" + uuid.uuid4().hex[:16]
        mapping = {value: f"{prefix}_{i}" for i, value in enumerate(sorted(known), 1)}
    wrapped = IdentifierClient(client, mapping)
    if not known.issubset(mapping):
        raise ValueError("Identifier map does not cover all declared case/slide identifiers")
    case_id = str(case.get("case_id") or case.get("id") or "")
    if not case_id or case_id not in mapping:
        raise ValueError("Identifier map does not include this case")
    target = Path(output_dir) / "llm_identifier_map.json"
    raw = (json.dumps({"schema": "pathxai-identifier-map-v1", "mapping": mapping},
                      ensure_ascii=False, indent=2) + "\n").encode()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return wrapped, {"mode": mode, "known_identifiers_replaced": True,
                     "mapping_path": str(target), "mapping_sha256": hashlib.sha256(raw).hexdigest(),
                     "identifier_count": len(mapping), "image_pixels_modified": False,
                     "scope": "Declared case/slide text identifiers only; local outputs retain original IDs."}
