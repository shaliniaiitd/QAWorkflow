```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Password reset test suite using pytest and Playwright.

Run with:
    pytest <script_name> -v
"""

import re
import pytest
from playwright.async_api import Page, expect

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


async def get_reset_link_from_email(email: str) -> str:
    """
    Mock function that simulates retrieving a password‑reset link from an email.
    In a real test environment this would query a test mailbox or a mock email
    service. Here we simply return a placeholder URL containing a token.
    """
    # In a real implementation, replace this with actual email retrieval logic.
    return f"https://example.com/reset-password?token=mocked-token-for-{email}"


async def is_email_sent(email: str) -> bool:
