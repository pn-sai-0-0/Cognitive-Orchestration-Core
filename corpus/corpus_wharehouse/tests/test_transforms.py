from _common.transforms import clean_text, classify_text, detect_language, quality_metrics


def test_clean_text_removes_tags_and_urls():
    text = "<p>Hello</p> visit https://example.com now"
    out = clean_text(text)
    assert "<p>" not in out
    assert "https://" not in out
    assert "Hello" in out


def test_detect_language_english():
    res = detect_language("This is a simple English sentence with the right stopwords and structure.")
    assert res["language"] == "en"
    assert res["confidence"] >= 0.55


def test_quality_metrics_scores_reasonable_text():
    res = quality_metrics("This is a clean paragraph with sensible words and enough content to evaluate quality fairly.")
    assert res["quality_score"] > 0.3


def test_classify_text_programming():
    res = classify_text("Python code defines a class and function implementing an algorithm.")
    assert "Programming" in res["labels"]
