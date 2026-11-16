"""Compatibility module for the Phase 1 workflow helpers.

The project’s tests expect these helper functions to live in
``src.workflow_phase1``. The implementation is maintained in
``src.qa_workflow`` and re-exported here for compatibility.
"""

from src.qa_workflow import _diagnose_failure_pattern, _is_valid_script_content, _strip_code_fences

__all__ = [
    "_diagnose_failure_pattern",
    "_is_valid_script_content",
    "_strip_code_fences",
]
