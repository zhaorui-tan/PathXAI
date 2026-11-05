# Request records for new runs

The default `trace_mode="record"` keeps two private files in the engine run directory:

| File | Contents |
| --- | --- |
| `llm_trace.jsonl` | Successful calls in the existing replay format: prompt, system/history messages, generation settings, image hashes and returned response. |
| `llm_trace.jsonl.events.jsonl` | Call start, request preparation, HTTP attempts, retry decisions, loaded responses, completion and failure. |

The event journal is flushed to disk before each observed HTTP attempt. A logging failure stops the call rather than sending an unrecorded request. A killed process may leave a start event without a terminal event; that is an incomplete call. No historical request is recovered by this feature.

HTTP observation covers the supplied synchronous `requests`/`urllib3` client. It records actual JSON payloads and replaces inline image data with hashes and byte counts. It omits HTTP headers, endpoint URLs and exception messages. Intermediate retry response bodies are not consumed; their status or exception type is recorded. Non-JSON bodies are hashed and streaming bodies are not consumed. API probes, other processes and unrelated HTTP libraries are outside this call journal.

These files contain clinical text and must stay with private run data. They are created with owner-only permissions. This is not a general secret scrubber: do not include credentials in prompts. Image paths are represented by hashes in the image-input list; identifiers appearing in prompt/history text remain unless the option below is enabled. Retain original images privately to verify hashes. Existing replay history can include inline image data supplied by a caller, so treat both files as sensitive.

## Optional replacement of known identifiers

Default behavior is unchanged. To replace declared case and slide identifiers in text sent to the VLM, set the following after `make_config(...)`:

```python
config["trace_mode"] = "record"
config["identifier_mode"] = "pseudonymize"
```

The adapter replaces the case ID and declared slide IDs/paths, including candidate metadata paths, with random `PXID_…` aliases before recording and transmission. It restores the original identifiers in the returned data used by the local pipeline. Clinical descriptions, known site, image pixels, scoring and the frozen scientific source are unchanged. The changed prompt can still affect model output; evaluating its effect requires new inference.

The private `llm_identifier_map.json` stores the exact replacements with owner-only permissions. Keep it with the run; never publish it as anonymized data. The resolved profile records its hash. Local reports, caches and other artifacts retain original IDs. This option does not detect undeclared identifiers in free text or remove identifiers embedded in image pixels.

Automatic maps reject short or numeric identifiers that could collide with clinical text. A reviewed map may be supplied through `identifier_map_path`; it must cover all declared identifiers. Its format is:

```json
{
  "schema": "pathxai-identifier-map-v1",
  "mapping": {"CASE-0042": "PXID_0123456789abcdef_1"}
}
```

Each original identifier needs a unique alias of that form. Replacement is literal and case-sensitive, longest matches first; review short identifiers carefully. Use a new output directory for every run.

## Replay

Keep the success trace and its `.events.jsonl` sibling together. New traces with a journal are checked for completed calls and matching response hashes. Failed or unfinished calls are rejected because failure replay is not implemented. Older traces without a journal retain the existing strict request-matching behavior; their receipt explicitly states that lifecycle evidence is unavailable. This check detects inconsistency, not deliberate deletion or forgery of audit files.

For a pseudonymized trace, also reuse the original private map:

```python
config["trace_mode"] = "replay"
config["trace_path"] = "/private/original-run/llm_trace.jsonl"
config["identifier_mode"] = "pseudonymize"
config["identifier_map_path"] = "/private/original-run/llm_identifier_map.json"
```

## Validation scope

`tests/test_request_audit.py` uses synthetic images/text and a localhost HTTP server. It exercises the frozen Qwen client's request construction, retries, errors, cancellation, identifier round trips and strict replay. It makes no external model requests. Live model and end-to-end GPU validation remain separate from these checks; historical study results do not change.
