# sql_generator.py
from __future__ import annotations
import os
import math
import pathlib
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np

UTC_NOW = lambda: datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ------------------------
# Helpers (identifier sanitization)
# ------------------------
import re


def snake_case(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip()
    name = re.sub(r"[\s\-/]+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.lower()
    if re.match(r"^[0-9]", name):
        name = "_" + name
    return name or "col"


def quote_ident(ident: str, dialect: str) -> str:
    """Quote identifier according to dialect. Use double quotes for PostgreSQL/SQLite."""
    # we assume ident is already snake_cased
    if dialect in ("postgresql", "sqlite"):
        return f'"{ident}"'
    # default: no quoting (but safe to use double quotes)
    return f'"{ident}"'


def quote_literal(value: Any) -> str:
    """Escape and quote a Python scalar value for SQL literal (single quotes)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NULL"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        # ensure proper formatting for NaN/inf
        if math.isfinite(float(value)):
            return repr(float(value))
        else:
            return "NULL"
    # booleans
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    # timestamps / datetimes
    if isinstance(value, (pd.Timestamp, datetime)):
        # format as ISO and quote
        return f"'{value.isoformat()}'"
    # fallback: string; escape single quotes by doubling them
    s = str(value)
    s = s.replace("'", "''")
    return f"'{s}'"


# ------------------------
# Dialect configuration
# ------------------------
# mapping function to choose SQL type for a pandas Series
def _choose_int_type(series: pd.Series, dialect: str) -> str:
    # choose BIGINT if values outside 32-bit signed range
    non_null = pd.to_numeric(series.dropna(), errors="coerce")
    if non_null.empty:
        # safe default
        return "INTEGER" if dialect == "sqlite" else "INTEGER"
    maxv = int(non_null.max())
    minv = int(non_null.min())
    if dialect == "postgresql":
        if maxv > 2_147_483_647 or minv < -2_147_483_648:
            return "BIGINT"
        else:
            return "INTEGER"
    else:
        # sqlite uses INTEGER (storage class handles sizes)
        return "INTEGER"


def _choose_float_type(series: pd.Series, dialect: str) -> str:
    if dialect == "postgresql":
        return "DOUBLE PRECISION"
    else:
        return "REAL"


DIALECTS = {
    "sqlite": {
        "quote_ident": quote_ident,
        "type_map": lambda series: (
            "INTEGER"
            if pd.api.types.is_integer_dtype(series.dropna())
            else ("REAL" if pd.api.types.is_float_dtype(series.dropna()) else ("BOOLEAN" if pd.api.types.is_bool_dtype(series.dropna()) else ("TIMESTAMP" if _is_datetime_series(series) else "TEXT")))
        ),
        "int_type_fn": _choose_int_type,
        "float_type_fn": _choose_float_type,
        "pk_autoinc": 'INTEGER PRIMARY KEY AUTOINCREMENT',
    },
    "postgresql": {
        "quote_ident": quote_ident,
        "type_map": lambda series: (
            ("BIGINT" if _choose_int_type(series, "postgresql") == "BIGINT" else "INTEGER")
            if pd.api.types.is_integer_dtype(series.dropna())
            else ("DOUBLE PRECISION" if pd.api.types.is_float_dtype(series.dropna()) else ("BOOLEAN" if pd.api.types.is_bool_dtype(series.dropna()) else ("TIMESTAMP" if _is_datetime_series(series) else "TEXT")))
        ),
        "int_type_fn": _choose_int_type,
        "float_type_fn": _choose_float_type,
        "pk_autoinc": 'SERIAL',
    },
}


# ------------------------
# Utilities for type inference
# ------------------------
def _is_datetime_series(s: pd.Series) -> bool:
    # reuse a small heuristic: dtype or parseable sample
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    sample = s.dropna().astype(str).head(20)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().sum() >= max(1, int(len(sample) / 2))


def infer_schema(df: pd.DataFrame, dialect: str) -> Dict[str, str]:
    """Return mapping column -> SQL type for given dialect."""
    mapping = {}
    dd = DIALECTS[dialect]
    for col in df.columns:
        series = df[col]
        # If mixed types (object with numeric strings), attempt to detect numeric
        # We conservatively treat ambiguous/mixed as TEXT
        # If pandas already has numeric dtype, use that
        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series) or pd.api.types.is_bool_dtype(series) or _is_datetime_series(series):
            # let dialect-specific function decide exact SQL type
            if pd.api.types.is_integer_dtype(series) or (pd.api.types.is_object_dtype(series) and _looks_like_int_series(series)):
                sql_type = dd["int_type_fn"](series, dialect) if pd.api.types.is_integer_dtype(series) else dd["int_type_fn"](series.astype(str), dialect)
            elif pd.api.types.is_float_dtype(series):
                sql_type = dd["float_type_fn"](series, dialect)
            elif pd.api.types.is_bool_dtype(series):
                sql_type = "BOOLEAN"
            elif _is_datetime_series(series):
                sql_type = "TIMESTAMP"
            else:
                sql_type = "TEXT"
        else:
            # object / mixed
            # test numeric parse
            if _looks_like_int_series(series):
                sql_type = dd["int_type_fn"](pd.to_numeric(series, errors="coerce"), dialect)
            elif _looks_like_float_series(series):
                sql_type = dd["float_type_fn"](pd.to_numeric(series, errors="coerce"), dialect)
            elif _is_datetime_series(series):
                sql_type = "TIMESTAMP"
            else:
                sql_type = "TEXT"
        mapping[col] = sql_type
    return mapping


def _looks_like_int_series(s: pd.Series) -> bool:
    # sample and test if most values parse as int
    sampled = s.dropna().astype(str).head(50)
    if sampled.empty:
        return False
    parsed = pd.to_numeric(sampled, errors="coerce")
    non_na = parsed.notna().sum()
    return non_na >= max(1, int(len(sampled) * 0.6)) and (parsed.dropna() % 1 == 0).all()


def _looks_like_float_series(s: pd.Series) -> bool:
    sampled = s.dropna().astype(str).head(50)
    if sampled.empty:
        return False
    parsed = pd.to_numeric(sampled, errors="coerce")
    non_na = parsed.notna().sum()
    return non_na >= max(1, int(len(sampled) * 0.6))


# ------------------------
# Core function
# ------------------------
def generate_sql(
    df: pd.DataFrame,
    dataset_id: str,
    dialect: str,
    table_name: Optional[str] = None,
    chunk_size: int = 10000,
    retain_in_memory_threshold: int = 100000,  # if rows <= this, we return insert SQL list; else write to file only
    sql_dir: str = "sql_exports",
) -> Dict[str, Any]:
    """
    Generate SQL CREATE TABLE and INSERT statements for the given dialect,
    streaming inserts to file in chunks.

    Returns dict:
      {
        "schema_sql": str,
        "insert_sql": List[str],  # may be empty for large datasets
        "file_path": str,
        "total_rows": int,
        "total_batches": int,
        "insert_in_file_only": bool
      }
    """
    dialect = dialect.lower()
    if dialect not in DIALECTS:
        raise ValueError(f"dialect must be one of {list(DIALECTS.keys())}")

    os.makedirs(sql_dir, exist_ok=True)
    if table_name is None:
        safe_table = f"dataset_{dataset_id.replace('-', '_')}"
    else:
        safe_table = table_name

    # sanitize column names and build mapping
    original_columns = list(df.columns)
    col_map = {c: snake_case(str(c)) for c in original_columns}
    # avoid duplicates after snake_case:
    seen = {}
    for orig, new in list(col_map.items()):
        if new in seen:
            # append index to make unique
            idx = 2
            candidate = f"{new}_{idx}"
            while candidate in seen.values():
                idx += 1
                candidate = f"{new}_{idx}"
            col_map[orig] = candidate
        seen[orig] = col_map[orig]

    # make a shallow copy with renamed columns for schema and iterating
    work_df = df.rename(columns=col_map).copy()

    dialect_conf = DIALECTS[dialect]
    col_types = infer_schema(work_df, dialect)

    # Build CREATE TABLE
    # Primary key logic
    pk_col = None
    if "id" in work_df.columns and (pd.api.types.is_integer_dtype(work_df["id"]) or _looks_like_int_series(work_df["id"])):
        pk_col = "id"
    else:
        # we'll create an auto-increment id if not present
        pk_col = "id"

    # Prepare CREATE TABLE lines
    lines = []
    q_table = dialect_conf["quote_ident"](safe_table, dialect)
    lines.append(f"-- Generated on {UTC_NOW()}")
    lines.append(f"-- Dataset: {dataset_id}")
    lines.append(f"CREATE TABLE {q_table} (")

    col_defs = []
    # if id exists in df and is chosen as pk, use its inferred type (and not autoinc for postgres if existing)
    if "id" in work_df.columns and pk_col == "id" and pd.api.types.is_integer_dtype(work_df["id"]):
        id_type = col_types["id"]
        if dialect == "postgresql":
            # if existing id col provided, keep as BIGINT/INTEGER but set as primary key (not SERIAL)
            col_defs.append(f"  {dialect_conf['quote_ident']('id', dialect)} {id_type} PRIMARY KEY")
        else:
            col_defs.append(f"  {dialect_conf['quote_ident']('id', dialect)} {id_type} PRIMARY KEY")
    else:
        # add auto-generated id first
        if dialect == "postgresql":
            col_defs.append(f"  {dialect_conf['quote_ident']('id', dialect)} {dialect_conf['pk_autoinc']} PRIMARY KEY")
        else:
            # sqlite: INTEGER PRIMARY KEY AUTOINCREMENT
            col_defs.append(f"  {dialect_conf['quote_ident']('id', dialect)} {dialect_conf['pk_autoinc']}")

    # Then other columns (skip id if we already used input id)
    for col in work_df.columns:
        if col == "id" and "id" in work_df.columns and pd.api.types.is_integer_dtype(work_df["id"]):
            # we already added id above
            continue
        sql_type = col_types.get(col, "TEXT")
        col_defs.append(f"  {dialect_conf['quote_ident'](col, dialect)} {sql_type}")

    lines.extend([",\n".join(col_defs)])
    lines.append(");")
    create_sql = "\n".join(lines)

    # Prepare INSERT generation
    total_rows = len(work_df)
    total_batches = math.ceil(total_rows / chunk_size) if chunk_size > 0 else 1
    file_path = os.path.join(sql_dir, f"{dataset_id}.sql")
    insert_sql_list: List[str] = []
    insert_in_file_only = total_rows > retain_in_memory_threshold

    # write to file incrementally
    with open(file_path, "w", encoding="utf-8") as fw:
        fw.write(create_sql + "\n\n")

        # build insert template
        columns_for_insert = [c for c in work_df.columns]
        q_columns = [dialect_conf["quote_ident"](c, dialect) for c in columns_for_insert]
        col_list_sql = ", ".join(q_columns)

        # iterate rows
        batch_values: List[str] = []
        batch_index = 0
        row_index = 0

        # Use itertuples which yields namedtuples; faster than iterrows
        for row in work_df.itertuples(index=False, name=None):
            # row is a tuple aligned with columns_for_insert
            vals = []
            for v in row:
                vals.append(quote_literal(v))
            row_sql = "(" + ", ".join(vals) + ")"
            batch_values.append(row_sql)
            row_index += 1

            if len(batch_values) >= chunk_size:
                batch_index += 1
                insert_stmt = f"INSERT INTO {q_table} ({col_list_sql}) VALUES\n" + ",\n".join(batch_values) + ";\n"
                fw.write(insert_stmt)
                if not insert_in_file_only:
                    insert_sql_list.append(insert_stmt)
                batch_values = []

        # final partial batch
        if batch_values:
            batch_index += 1
            insert_stmt = f"INSERT INTO {q_table} ({col_list_sql}) VALUES\n" + ",\n".join(batch_values) + ";\n"
            fw.write(insert_stmt)
            if not insert_in_file_only:
                insert_sql_list.append(insert_stmt)

    # logging-like return
    return {
        "schema_sql": create_sql,
        "insert_sql": insert_sql_list,
        "file_path": file_path,
        "total_rows": total_rows,
        "total_batches": total_batches,
        "dialect": dialect,
        "insert_in_file_only": insert_in_file_only,
    }
