"""Case defaults, saved selections and exact source-block remapping."""

import copy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from .common import ROOT, RUNTIME, atomic, read, clock_block


class Cases:
    def __init__(self, settings_path=None, catalog_path=None):
        catalog = Path(
            catalog_path or os.environ.get("DUO_CASE_CATALOG", ROOT / "configs/cases/catalog.json")
        )
        self.catalog_path = catalog.resolve()
        self.items = {c["id"]: c for c in read(self.catalog_path)}
        self.custom_catalog = RUNTIME / "custom-cases" / "catalog.json"
        if self.custom_catalog.is_file():
            self.items.update({c["id"]: c for c in read(self.custom_catalog)})
        self.path = settings_path or RUNTIME / "case_settings.json"
        self.saved = read(self.path) if self.path.exists() else {}

    def create(self, label, description, reference):
        from PIL import Image, ImageOps

        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 60:
            raise ValueError("Enter a Case name between 1 and 60 characters.")
        self.validate_description(description)
        try:
            with Image.open(BytesIO(reference), formats=["JPEG", "PNG", "WEBP"]) as image:
                if image.width * image.height > 20_000_000:
                    raise ValueError("Reference images must be at most 20 megapixels.")
                image = ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Choose a valid JPEG, PNG, or WebP reference image.") from exc
        image.thumbnail((2048, 2048))
        case_id = "custom-" + uuid4().hex
        image_path = self.custom_catalog.parent / f"{case_id}.jpg"
        case = dict(
            id=case_id,
            label=label.strip(),
            subtitle="",
            speakers={"assistant": "Assistant", "user": "User"},
            description=description.strip(),
            opening="",
            opening_chunks=[],
            assistant_image=str(image_path),
        )
        self.custom_catalog.parent.mkdir(parents=True, exist_ok=True)
        entries = read(self.custom_catalog) if self.custom_catalog.exists() else []
        try:
            image.save(image_path, "JPEG", quality=95)
            atomic(self.custom_catalog, [*entries, case])
        except OSError:
            image_path.unlink(missing_ok=True)
            raise
        self.items[case_id] = case
        return case_id

    def frontend(self):
        return [
            dict(
                id=case["id"],
                label=case["label"],
                subtitle=case.get("subtitle", ""),
                description=case["description"],
                assistant_image=f"/api/cases/{quote(case['id'], safe='')}/reference-image",
            )
            for case in self.items.values()
        ]

    @staticmethod
    def validate_description(text):
        if not isinstance(text, str) or not text.strip() or len(text) > 6000:
            raise ValueError("Enter an assistant description of up to 6000 characters.")
        if any(t in text for t in ("<|", "<wait>", "<blank>", "<chunk_")):
            raise ValueError("Assistant description contains intent control tokens")

    def default(self, case_id):
        if case_id not in self.items:
            raise ValueError("Unknown Case: " + str(case_id))
        case = self.items[case_id]
        return dict(
            case_id=case_id,
            description=case["description"],
            opening_chunk_ids=[c["source_index"] for c in case["opening_chunks"]],
        )

    def settings(self, case_id, values=None):
        default = self.default(case_id)
        result = dict(self.saved.get(case_id, default))
        if values is not None:
            for key in ("description", "opening_chunk_ids"):
                if key in values:
                    result[key] = values[key]
        ids = result["opening_chunk_ids"]
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            raise ValueError("Opening chunks must be a list of source indices")
        if ids != sorted(set(ids)) or not set(ids) <= set(default["opening_chunk_ids"]):
            raise ValueError("Opening chunks must be unique and keep source order")
        self.validate_description(result["description"])
        result["case_id"] = case_id
        result["config_id"] = fingerprint(result)
        return copy.deepcopy(result)

    def save(self, settings):
        self.saved[settings["case_id"]] = copy.deepcopy(settings)
        atomic(self.path, self.saved)

    def public(self):
        result = []
        for case in self.items.values():
            public = copy.deepcopy(case)
            public["assistant_image"] = self.assistant_image_url(case["id"])
            public["settings"] = self.settings(case["id"])
            result.append(public)
        return result

    def assistant_image(self, case_id):
        self.default(case_id)
        value = Path(self.items[case_id]["assistant_image"])
        path = value if value.is_absolute() else ROOT / value
        if not path.is_file():
            raise FileNotFoundError(f"Assistant reference image not found: {path}")
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError(f"Unsupported assistant reference image: {path}")
        return path.resolve()

    def assistant_image_url(self, case_id):
        self.assistant_image(case_id)
        return f"/api/cases/{quote(case_id, safe='')}/reference-image"


def fingerprint(settings):
    data = {key: settings[key] for key in ("case_id", "description", "opening_chunk_ids")}
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]


def remap_opening(source, source_ids):
    """Keep original token IDs and their relative commit times with the video block."""
    record = copy.deepcopy(source)
    canonical = record["canonical"]
    tokens, attachments, mapping = [], [], []
    for local, index in enumerate(source_ids):
        block = clock_block(index)
        offset = (local - index) * 17 * 1000 / 24
        mapping.append(
            dict(
                local=local,
                source_index=index,
                frame_start=block["frame_start"],
                frame_stop=block["frame_stop"],
            )
        )
        for token in source["canonical"]["text_tokens"]:
            if (
                token["role"] != "user"
                or not block["frame_start"] * 1000
                <= token["commit_ms"] * 24
                < block["frame_stop"] * 1000
            ):
                continue
            copied = dict(token, source_chunk=index, source_commit_ms=token["commit_ms"])
            for key in ("start_ms", "end_ms", "commit_ms"):
                copied[key] = token[key] + offset
            tokens.append(copied)
        for item in source["canonical"].get("emotions", {}).get("attachments", []):
            stamp = item.get("commit_ms")
            if (
                item.get("role") == "user"
                and stamp is not None
                and block["start_ms"] <= stamp < block["end_ms"]
            ):
                attachments.append(dict(item, commit_ms=stamp + offset, source_chunk=index))
    canonical["text_tokens"] = tokens
    canonical["emotions"] = dict(attachments=attachments)
    canonical["fine_alignment"] = dict(turns=[])
    record["opening_source_mapping"] = mapping
    return record
