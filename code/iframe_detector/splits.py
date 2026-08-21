from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SPLIT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "splits_recovered_20260727_bidir.json"


def load_split_config(path: Path | str | None = DEFAULT_SPLIT_CONFIG_PATH) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _norm_name(value: str) -> str:
    value = str(value).strip().lower().replace("\\", "/")
    name = value.rsplit("/", 1)[-1]
    if name.endswith(".pcapng"):
        name = name[:-7]
    elif name.endswith(".pcap"):
        name = name[:-5]
    return name


def capture_records(config: dict | None) -> list[dict]:
    return list(config.get("captures", [])) if config else []


def capture_by_id(config: dict | None) -> dict[str, dict]:
    return {str(row["capture_id"]): row for row in capture_records(config)}


def alias_to_capture_id(config: dict | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for row in capture_records(config):
        capture_id = str(row["capture_id"])
        names = [capture_id, row.get("pcap_filename", ""), *row.get("aliases", [])]
        for name in names:
            if not name:
                continue
            key = _norm_name(str(name))
            previous = aliases.get(key)
            if previous and previous != capture_id:
                raise ValueError(f"split config alias {key!r} maps to both {previous!r} and {capture_id!r}")
            aliases[key] = capture_id
    return aliases


def canonical_capture_id(value: str, config: dict | None) -> str:
    if not config:
        return str(value)
    return alias_to_capture_id(config).get(_norm_name(value), str(value))


def resolve_capture_record(value: str | Path, config: dict | None) -> dict | None:
    if not config:
        return None
    capture_id = canonical_capture_id(str(value), config)
    return capture_by_id(config).get(capture_id)


def canonicalize_capture_columns(df: pd.DataFrame, config: dict | None, drop_unassigned: bool = False) -> pd.DataFrame:
    if df.empty or not config or "capture_id" not in df.columns:
        return df.copy()
    records = capture_by_id(config)
    out = df.copy()
    out["capture_id"] = out["capture_id"].map(lambda x: canonical_capture_id(str(x), config))
    out["split_role"] = out["capture_id"].map(lambda x: records.get(str(x), {}).get("role", "unassigned"))
    out["split_id"] = str(config.get("split_id", ""))
    if "application" in out.columns:
        out["application"] = out["capture_id"].map(
            lambda x: records.get(str(x), {}).get("application")
        ).fillna(out["application"])
    else:
        out["application"] = out["capture_id"].map(lambda x: records.get(str(x), {}).get("application", "unknown"))
    if drop_unassigned:
        out = out[out["split_role"].ne("unassigned")].copy()
    return out


def split_masks(
    df: pd.DataFrame,
    train_roles: Iterable[str] = ("development",),
    test_roles: Iterable[str] = ("final_app_ood",),
) -> tuple[pd.Series, pd.Series]:
    if "split_role" not in df.columns:
        empty = pd.Series(False, index=df.index)
        return empty, empty
    train_set = {str(x) for x in train_roles}
    test_set = {str(x) for x in test_roles}
    role = df["split_role"].astype(str)
    return role.isin(train_set), role.isin(test_set)


def summarize_split_table(df: pd.DataFrame) -> dict:
    if df.empty or "capture_id" not in df.columns:
        return {"row_count": int(len(df)), "captures": [], "roles": {}}
    roles = {}
    if "split_role" in df.columns:
        roles = {str(k): int(v) for k, v in df["split_role"].value_counts(dropna=False).sort_index().items()}
    captures = []
    group_cols = ["capture_id"]
    if "application" in df.columns:
        group_cols.append("application")
    if "split_role" in df.columns:
        group_cols.append("split_role")
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"capture_id": str(keys[0]), "row_count": int(len(group))}
        if "application" in group_cols:
            row["application"] = str(keys[group_cols.index("application")])
        if "split_role" in group_cols:
            row["split_role"] = str(keys[group_cols.index("split_role")])
        captures.append(row)
    return {"row_count": int(len(df)), "roles": roles, "captures": captures}
