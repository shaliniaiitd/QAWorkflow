'''Test cases for workflow_phase1 helpers.'''


from src.workflow_phase1 import (
    _diagnose_failure_pattern,
    _is_valid_script_content,
    _strip_code_fences,
)


def test_is_valid_script_content_empty_string():
    assert not _is_valid_script_content("")


def test_is_valid_script_content_under_50_characters():
    assert not _is_valid_script_content("def test_short():\n    pass\n")


def test_is_valid_script_content_valid_test_script():
    content = "def test_login_flow():\n    " + "pass  # " + "x" * 50
    assert _is_valid_script_content(content)


def test_is_valid_script_content_without_test_function():
    content = "# " + "x" * 60
    assert not _is_valid_script_content(content)


def test_strip_code_fences_with_python_fence():
    fenced = "```python\nprint('hello')\n```"
    assert _strip_code_fences(fenced) == "print('hello')"


def test_strip_code_fences_without_fences():
    plain = "print('hello')"
    assert _strip_code_fences(plain) == plain


def test_strip_code_fences_with_only_opening_fence():
    opening_only = "```python\nprint('hello')"
    assert _strip_code_fences(opening_only) == "print('hello')"


def test_diagnose_failure_pattern_selector_xpath():
    result = _diagnose_failure_pattern("ElementNotFoundError: xpath selector was not found")
    assert "XPath" in result


def test_diagnose_failure_pattern_selector_css():
    result = _diagnose_failure_pattern("ElementNotFoundError: css selector was not found")
    assert "CSS" in result


def test_diagnose_failure_pattern_timeout():
    result = _diagnose_failure_pattern("TimeoutError: waiting for selector exceeded the limit")
    assert "Timing/Wait" in result


def test_diagnose_failure_pattern_assertion():
    result = _diagnose_failure_pattern("AssertionError: expected 'Welcome', actual 'Goodbye'")
    assert "Assertion" in result


def test_diagnose_failure_pattern_navigation():
    result = _diagnose_failure_pattern("NavigationError: page.goto returned HTTP 404")
    assert "Navigation/Page Load" in result


def test_diagnose_failure_pattern_fallback():
    result = _diagnose_failure_pattern("Quantum flux anomaly observed")
    assert "Unknown Failure" in result
