# data_cleaner.py
from __future__ import annotations
import re
import hashlib
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime, timezone
import numpy as np
import pandas as pd

JSON_TIMESTAMP = lambda: datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _now():
    return JSON_TIMESTAMP()


def _sample_before_after(series: pd.Series, n: int = 3):
    before = series.dropna().astype(str).head(n).tolist()
    after = series.dropna().astype(str).head(n).tolist()
    return {"before": before, "after": after}


def _is_datetime_series(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    # try to parse small sample
    sample = s.dropna().astype(str).head(10)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().sum() >= max(1, int(len(sample) / 2))


def snake_case(name: str) -> str:
    # ascii-safe snake_case: remove/replace special chars, spaces -> _
    if not isinstance(name, str):
        name = str(name)
    # normalize whitespace and hyphens, remove special unicode chars by ascii fallback
    name = name.strip()
    # replace spaces and separators with underscore
    name = re.sub(r"[\\s\\-\\/]+", "_", name)
    # replace non-alphanumeric / underscore with underscore
    name = re.sub(r"[^0-9A-Za-z_]", "_", name)
    # collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # lower
    name = name.lower()
    # ensure doesn't start with digit
    if re.match(r"^[0-9]", name):
        name = "_" + name
    return name or "col"


def _hash_value(v: str) -> str:
    return hashlib.sha256(v.encode("utf-8")).hexdigest()


def _mask_email(v: str, mode: str = "mask") -> str:
    # simple email mask/hash
    if mode == "drop":
        return np.nan
    if mode == "hash":
        return _hash_value(v.lower())
    # mask: keep domain, mask user part to first char + ***
    parts = v.split("@")
    if len(parts) != 2:
        return v
    user, domain = parts
    if len(user) <= 1:
        masked = "*" * len(user)
    else:
        masked = user[0] + ("*" * (max(1, len(user) - 1)))
    return masked + "@" + domain


def _mask_phone(v: str, mode: str = "mask") -> str:
    if mode == "drop":
        return np.nan
    if mode == "hash":
        return _hash_value(re.sub(r"\\D", "", v))
    # mask middle digits, keep last 3 or 4
    digits = re.sub(r"\\D", "", v)
    if len(digits) <= 4:
        return "*" * len(digits)
    keep = 4
    masked = ("*" * (len(digits) - keep)) + digits[-keep:]
    return masked


def _mask_credit_card(v: str, mode: str = "mask") -> str:
    if mode == "drop":
        return np.nan
    digits = re.sub(r"\\D", "", str(v))
    if mode == "hash":
        return _hash_value(digits)
    if len(digits) <= 4:
        return "*" * len(digits)
    return ("*" * (len(digits) - 4)) + digits[-4:]


def _mask_name(v: str, mode: str = "mask") -> str:
    if pd.isna(v):
        return v
    if mode == "drop":
        return np.nan
    if mode == "hash":
        return _hash_value(v)
    # mask name: initials e.g., "John Doe" -> "J. D."
    parts = re.split(r"\\s+", str(v).strip())
    initials = [p[0].upper() + "." for p in parts if p]
    return " ".join(initials)


def detect_pii_in_series(s: pd.Series) -> Dict[str, Any]:
    # returns dict of flags and example matches
    out = {"email": False, "phone": False, "credit_card": False, "name_like": False, "examples": {}}
    sample = s.dropna().astype(str).head(50)
    if sample.empty:
        return out
    email_re = re.compile(r"[^\\s@]+@[^\\s@]+\\.[^\\s@]+")
    phone_re = re.compile(r"(\\+?\\d[\\d\\-\\s\\(\\)]{6,}\\d)")
    cc_re = re.compile(r"(?:\\d[ -]*?){13,19}")
    name_like = 0
    for v in sample:
        if email_re.search(v):
            out["email"] = True
            out["examples"].setdefault("email", []).append(v)
        if phone_re.search(v):
            out["phone"] = True
            out["examples"].setdefault("phone", []).append(v)
        if cc_re.search(v):
            out["credit_card"] = True
            out["examples"].setdefault("credit_card", []).append(v)
        # naive name heuristic: alphabetic tokens and capitalization
        tokens = re.findall(r"[A-Za-z]{2,}", v)
        if len(tokens) >= 2 and len(v) < 60 and v.strip()[0].isalpha():
            name_like += 1
    out["name_like"] = name_like >= max(1, int(len(sample) * 0.2))
    return out


def infer_sql_type(series: pd.Series) -> str:
    # Map pandas series to SQL-like types
    if pd.api.types.is_integer_dtype(series.dropna()):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series.dropna()):
        return "FLOAT"
    if _is_datetime_series(series):
        # check if contains time
        if pd.api.types.is_datetime64_any_dtype(series):
            # check if any has time not 00:00:00
            sample = series.dropna().head(100)
            if sample.empty:
                return "DATE"
            parsed = pd.to_datetime(sample, errors="coerce")
            # if any time component non-zero => DATETIME
            if (parsed.dt.hour.fillna(0) + parsed.dt.minute.fillna(0) + parsed.dt.second.fillna(0)).any():
                return "DATETIME"
            return "DATE"
        else:
            return "DATE"
    # fallback: text
    return "TEXT"


def _safe_cast_series(series: pd.Series, target: str) -> Tuple[pd.Series, bool]:
    """Attempt to cast series to target type. Return (series, success_flag)."""
    try:
        if target == "INTEGER":
            casted = pd.to_numeric(series, errors="coerce").dropna().astype("Int64")
            # If many coerced to NaN then consider failure
            return pd.to_numeric(series, errors="coerce").astype("Int64"), True
        if target == "FLOAT":
            return pd.to_numeric(series, errors="coerce").astype(float), True
        if target in ("DATE", "DATETIME"):
            parsed = pd.to_datetime(series, errors="coerce")
            return parsed, True
        # TEXT
        return series.astype(str), True
    except Exception:
        return series, False


def clean_dataframe(df: pd.DataFrame, config: Optional[Dict] = None) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Clean a pandas DataFrame and return (clean_df, audit_log)

    config keys (defaults):
     - numeric_missing_threshold: 0.2
     - numeric_skew_threshold: 1.0  (if abs(skew) > this -> prefer median)
     - datetime_missing_threshold: 0.1
     - negative_policy: 'abs' | 'nan' | 'none'
     - pii_mode: 'mask' | 'hash' | 'drop'
     - drop_column_missing_threshold: 0.9  (if > this missing, drop)
    """
    cfg = {
        "numeric_missing_threshold": 0.2,
        "numeric_skew_threshold": 1.0,
        "datetime_missing_threshold": 0.1,
        "negative_policy": "abs",
        "pii_mode": "mask",
        "drop_column_missing_threshold": 0.9,
    }
    if config:
        cfg.update(config)

    audit: List[Dict] = []
    # Work on a copy
    result = df.copy(deep=True)

    # 1. Column Normalization
    old_cols = list(result.columns)
    new_cols = [snake_case(c) for c in old_cols]
    rename_map = dict(zip(old_cols, new_cols))
    result.rename(columns=rename_map, inplace=True)
    audit.append({
        "step": "column_normalization",
        "timestamp": _now(),
        "columns_before": old_cols,
        "columns_after": new_cols,
        "cols_renamed": rename_map,
        "counts": {"num_columns_before": len(old_cols), "num_columns_after": len(new_cols)},
    })

    # 2. Duplicate Removal
    before = len(result)
    result = result.drop_duplicates(ignore_index=True)
    after = len(result)
    removed = before - after
    audit.append({
        "step": "duplicate_removal",
        "timestamp": _now(),
        "counts": {"rows_before": before, "rows_after": after, "duplicates_removed": removed},
    })

    # 3. Missing Value Handling
    mv_log = {"step": "missing_value_handling", "timestamp": _now(), "columns": {}, "counts": {}}
    for col in result.columns:
        s = result[col]
        n_total = len(s)
        n_missing = s.isna().sum()
        pct_missing = n_missing / max(1, n_total)
        col_info = {"missing": int(n_missing), "pct_missing": float(pct_missing)}
        # skip if all missing
        if n_missing == n_total:
            # drop or leave - config can be extended; for now, leave but mark
            col_info["action"] = "all_missing_left"
            mv_log["columns"][col] = col_info
            continue

        # numeric
        if pd.api.types.is_numeric_dtype(s):
            if pct_missing < cfg["numeric_missing_threshold"]:
                skew = float(s.dropna().skew()) if s.dropna().shape[0] > 2 else 0.0
                if abs(skew) > cfg["numeric_skew_threshold"]:
                    fill_val = float(s.dropna().median())
                    method = "median"
                else:
                    fill_val = float(s.dropna().mean())
                    method = "mean"
                result[col] = s.fillna(fill_val)
                col_info.update({"action": "imputed", "method": method, "imputed_value": fill_val})
            else:
                # >= threshold
                if pct_missing >= cfg["drop_column_missing_threshold"]:
                    result.drop(columns=[col], inplace=True)
                    col_info.update({"action": "dropped_due_to_missing", "reason": "high_missing"})
                else:
                    # fill with mode if available else leave
                    mode_val = s.mode().iloc[0] if not s.mode().empty else np.nan
                    result[col] = s.fillna(mode_val)
                    col_info.update({"action": "imputed", "method": "mode", "imputed_value": mode_val})
        # categorical / object
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            if n_missing > 0:
                mode_val = s.mode().iloc[0] if not s.mode().empty else ""
                result[col] = s.fillna(mode_val)
                col_info.update({"action": "imputed", "method": "mode", "imputed_value": mode_val})
        # datetime
        elif _is_datetime_series(s):
            parsed = pd.to_datetime(s, errors="coerce")
            n_missing_dt = parsed.isna().sum()
            pct_missing_dt = n_missing_dt / max(1, n_total)
            if pct_missing_dt < cfg["datetime_missing_threshold"]:
                median_date = parsed.dropna().median()
                result[col] = parsed.fillna(median_date)
                col_info.update({"action": "imputed_datetime", "imputed_value": str(median_date)})
            else:
                # set to NaT (coerce)
                result[col] = parsed
                col_info.update({"action": "set_nat_due_to_missing"})
        else:
            # fallback: leave as-is
            col_info.update({"action": "no_action"})
        mv_log["columns"][col] = col_info
    audit.append(mv_log)

    # 4. Anomaly Detection & Correction
    anom_log = {"step": "anomaly_detection", "timestamp": _now(), "columns": {}, "counts": {}}
    for col in list(result.columns):
        s = result[col]
        col_changes = {"corrections": 0, "examples": {}}
        # datetime impossible dates
        if _is_datetime_series(s):
            parsed = pd.to_datetime(s, errors="coerce")
            today = pd.Timestamp.now(tz=None)
            upper = today + pd.DateOffset(years=1)
            lower = pd.Timestamp(year=1900, month=1, day=1)
            mask_bad = (parsed > upper) | (parsed < lower)
            n_bad = int(mask_bad.sum())
            if n_bad > 0:
                parsed.loc[mask_bad] = pd.NaT
                result[col] = parsed
                col_changes["corrections"] += n_bad
                col_changes["examples"]["bad_dates_sample"] = parsed[mask_bad].head(3).astype(str).tolist()
        # numeric outliers
        if pd.api.types.is_numeric_dtype(s):
            arr = pd.to_numeric(s, errors="coerce")
            mean = float(arr.mean(skipna=True)) if not np.isnan(arr.mean()) else 0.0
            std = float(arr.std(skipna=True)) if not np.isnan(arr.std()) and arr.std() != 0 else 0.0
            if std > 0:
                cap_low = mean - 5 * std
                cap_high = mean + 5 * std
                mask_low = arr < cap_low
                mask_high = arr > cap_high
                n_caps = int(mask_low.sum() + mask_high.sum())
                if n_caps > 0:
                    arr.loc[mask_low] = cap_low
                    arr.loc[mask_high] = cap_high
                    result[col] = arr
                    col_changes["corrections"] += n_caps
                    col_changes["examples"]["outlier_samples"] = arr[mask_high | mask_low].head(3).tolist()
        # negative handling - heuristic: if column name includes words like 'amount', 'price', 'qty', treat as positive
        if pd.api.types.is_numeric_dtype(s):
            name = col.lower()
            positive_like = any(k in name for k in ("amount", "price", "qty", "quantity", "count", "age"))
            if positive_like:
                if cfg["negative_policy"] == "abs":
                    # convert negatives to abs
                    arr = pd.to_numeric(result[col], errors="coerce")
                    mask_neg = arr < 0
                    n_neg = int(mask_neg.sum())
                    if n_neg > 0:
                        arr.loc[mask_neg] = arr.loc[mask_neg].abs()
                        result[col] = arr
                        col_changes["corrections"] += n_neg
                        col_changes["examples"].setdefault("negatives_fixed_sample", arr[mask_neg].head(3).tolist())
                elif cfg["negative_policy"] == "nan":
                    arr = pd.to_numeric(result[col], errors="coerce")
                    mask_neg = arr < 0
                    n_neg = int(mask_neg.sum())
                    if n_neg > 0:
                        arr.loc[mask_neg] = np.nan
                        result[col] = arr
                        col_changes["corrections"] += n_neg
                        col_changes["examples"].setdefault("negatives_set_nan_sample", arr[mask_neg].head(3).tolist())
        if col_changes["corrections"] > 0:
            anom_log["columns"][col] = col_changes
    audit.append(anom_log)

    # 5. PII Detection & Masking
    pii_log = {"step": "pii_detection_masking", "timestamp": _now(), "columns": {}, "pii_mode": cfg["pii_mode"]}
    for col in list(result.columns):
        s = result[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            # skip non-string-like
            continue
        flags = detect_pii_in_series(s)
        if not any((flags["email"], flags["phone"], flags["credit_card"], flags["name_like"])):
            continue
        action = {"detected": flags, "masked_count": 0, "method": cfg["pii_mode"]}
        # perform masking based on detections
        def _apply_mask(v):
            if pd.isna(v):
                return v
            vstr = str(v)
            if flags["email"] and re.search(r"[^\\s@]+@[^\\s@]+\\.[^\\s@]+", vstr):
                return _mask_email(vstr, cfg["pii_mode"])
            if flags["credit_card"] and re.search(r"(?:\\d[ -]*?){13,19}", vstr):
                return _mask_credit_card(vstr, cfg["pii_mode"])
            if flags["phone"] and re.search(r"(\\+?\\d[\\d\\-\\s\\(\\)]{6,}\\d)", vstr):
                return _mask_phone(vstr, cfg["pii_mode"])
            if flags["name_like"]:
                return _mask_name(vstr, cfg["pii_mode"])
            return v
        before_sample = s.head(3).astype(str).tolist()
        result[col] = s.apply(_apply_mask)
        after_sample = result[col].head(3).astype(str).tolist()
        action["masked_count"] = int(s.apply(lambda x: (x != result[col][s.index[s == x][0]]).any() if False else 0))  # placeholder 0
        # We can't easily compute masked_count without elementwise compare (we can do it now)
        cmp = s.astype(str).fillna("##NA##") != result[col].astype(str).fillna("##NA##")
        action["masked_count"] = int(cmp.sum())
        action["examples"] = {"before": before_sample, "after": after_sample}
        pii_log["columns"][col] = action
    audit.append(pii_log)

    # 6. Schema Normalization for SQL
    schema_log = {"step": "schema_normalization", "timestamp": _now(), "columns": {}, "counts": {}}
    for col in list(result.columns):
        series = result[col]
        inferred = infer_sql_type(series)
        casted_series, success = _safe_cast_series(series, inferred)
        if success:
            result[col] = casted_series
            schema_log["columns"][col] = {"sql_type": inferred, "cast_success": True}
        else:
            # fallback to TEXT
            result[col] = series.astype(str)
            schema_log["columns"][col] = {"sql_type": "TEXT", "cast_success": False}
    audit.append(schema_log)

    # Final audit summary counts
    audit.append({
        "step": "summary",
        "timestamp": _now(),
        "rows": int(len(result)),
        "columns": list(result.columns),
    })

    return result, audit
