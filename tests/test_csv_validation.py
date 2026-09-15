"""
Unit Tests for CSV Batch Validation and Processing.
"""

import io
import pytest
import pandas as pd


def validate_and_parse_csv(file_obj) -> pd.DataFrame:
    """Helper validator matching Streamlit app CSV loader."""
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Invalid CSV format: {str(e)}")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Check for review column case-insensitively
    matching_cols = [col for col in df.columns if col.strip().lower() == "review"]
    if not matching_cols:
        raise ValueError("Uploaded CSV must contain a 'review' column.")

    review_col = matching_cols[0]
    df["review"] = df[review_col].fillna("").astype(str)
    
    # Filter rows that have at least some alphanumeric content
    valid_rows = df[df["review"].str.strip().str.len() > 0]
    if valid_rows.empty:
        raise ValueError("The CSV contains no valid non-empty review text rows.")

    return valid_rows


def test_valid_csv():
    csv_data = "review\nGreat camera and fast shipping!\nTerrible battery life.\nNeutral statement."
    file_obj = io.StringIO(csv_data)
    df = validate_and_parse_csv(file_obj)
    assert len(df) == 3
    assert "review" in df.columns


def test_case_insensitive_review_column():
    csv_data = "Review,Rating\nSuperb laptop!,5\nBroken screen,1"
    file_obj = io.StringIO(csv_data)
    df = validate_and_parse_csv(file_obj)
    assert len(df) == 2


def test_missing_review_column_raises_error():
    csv_data = "feedback_text,user_id\nLoved it,101"
    file_obj = io.StringIO(csv_data)
    with pytest.raises(ValueError, match="Uploaded CSV must contain a 'review' column."):
        validate_and_parse_csv(file_obj)


def test_empty_csv_raises_error():
    csv_data = ""
    file_obj = io.StringIO(csv_data)
    with pytest.raises(ValueError):
        validate_and_parse_csv(file_obj)


def test_csv_with_all_empty_reviews_raises_error():
    csv_data = "review,other_col\n   ,1\n   ,2"
    file_obj = io.StringIO(csv_data)
    with pytest.raises(ValueError, match="The CSV contains no valid non-empty review text rows."):
        validate_and_parse_csv(file_obj)
