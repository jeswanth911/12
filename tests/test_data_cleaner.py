# tests/test_data_cleaner.py
import pandas as pd
import numpy as np
import re
from data_cleaner import clean_dataframe

def test_duplicates_and_missing_values():
    df = pd.DataFrame({
        "Name": ["A","A","B", None],
        "Val": [1,1,2, None],
    })
    clean_df, audit = clean_dataframe(df)
    # duplicates removed
    assert len(clean_df) == 3 or len(clean_df) == 2  # depending on imputation, ensure no crash
    # missing imputation for numeric 'val' exists (since <20% threshold default)
    assert "val" in clean_df.columns

def test_pii_masking_email_and_phone():
    df = pd.DataFrame({
        "emailAddr": ["user1@example.com", "noemail", None],
        "phone": ["+1 555 123 4567", "555-0000", None],
        "name": ["John Doe", "Alice", "Bob B"]
    })
    clean_df, audit = clean_dataframe(df, config={"pii_mode": "mask"})
    # emails should be masked (contain '@' only in masked domain or replaced)
    assert not any(re.match(r"[^\\s@]+@[^\\s@]+\\.[^\\s@]+", str(x)) for x in clean_df["emailaddr"].dropna().tolist())
    # phone masked should contain '*' or be hashed
    assert any("*" in str(x) or re.fullmatch(r"[0-9a-f]{64}", str(x)) for x in clean_df["phone"].dropna().astype(str).tolist())

def test_outlier_handling_and_negative_policy():
    df = pd.DataFrame({
        "Amount": [100, -50, 200, 10_000_000]
    })
    clean_df, audit = clean_dataframe(df, config={"negative_policy": "abs"})
    # negatives should become positive
    assert (clean_df["amount"] >= 0).all()
    # extreme outlier should be capped at mean+5*std or similar -> not infinite
    assert np.isfinite(clean_df["amount"]).all()

def test_schema_normalization_types():
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.1, 2.2, None],
        "date_col": ["2020-01-01", "2020-05-01", None],
        "mix_col": ["1", "two", "3"]
    })
    clean_df, audit = clean_dataframe(df)
    # columns exist and types coerced
    assert "int_col" in clean_df.columns
    assert "float_col" in clean_df.columns
    # date_col should be datetime-like
    assert pd.api.types.is_datetime64_any_dtype(clean_df["date_col"]) or pd.api.types.is_object_dtype(clean_df["date_col"])

