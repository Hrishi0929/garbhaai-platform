"""
Streamlit apps aren't practically unit-testable end-to-end without a
running browser, so these tests cover the pure helper logic in main.py
(the parts with actual behavior worth protecting) rather than the
Streamlit rendering itself.
"""
from main import summarize_features


def test_summarize_features_short_vector():
    out = summarize_features([1.0, 2.0])
    assert "2 dims total" in out


def test_summarize_features_truncates_preview():
    features = [float(i) for i in range(512)]
    out = summarize_features(features)
    assert "512 dims total" in out
    # only the first 5 values should appear in the preview text
    assert "0.000, 1.000, 2.000, 3.000, 4.000" in out
