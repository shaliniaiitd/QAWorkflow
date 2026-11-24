import pytest
from playwright.sync_api import expect

BASE_URL = "https://shaliniaiitd.github.io"

@pytest.fixture(autouse=True)
def base_url(page):
    page.goto(BASE_URL)

# Happy Path
def test_successful_password_reset_flow(page):
    """Scenario: Successful password reset flow"""
    # Go to Forgot Password page
    page.goto(f"{BASE_URL}/forgot-password")
    # Enter registered email
    page.fill("input[name='email']", "user@example.com")
    # Click Send Reset Link
    page.click("button:has-text('Send Reset Link')")
    # Confirmation message
    expect(page.locator("text=If an account with that email exists, a reset link has been sent.")).to_be_visible()
    # Email delivery is out of scope; skip
    pytest.skip("Email delivery verification not implemented")
    # Click reset link (simulate by navigating to reset page)
    page.goto(f"{BASE_URL}/reset-password?token=validtoken")
    # Verify Reset Password page
    expect(page.locator("h1:has-text('Reset Password')")).to_be_visible()
    # Verify fields
    expect(page.locator("input[name='newPassword']")).to_be_visible()
    expect(page.locator("input[name='confirmPassword']")).to_be_visible()
    # Enter valid passwords
    page.fill("input[name='newPassword']", "StrongPass!23")
    page.fill("input[name='confirmPassword']", "StrongPass!23")
    # Click Reset Password
    page.click("button:has-text('Reset Password')")
    # Redirect to login page
    expect(page).to_have_url(f"{BASE_URL}/login")
    # Success notification
    expect(page.locator("text=Your password has been reset successfully.")).to_be_visible()

# Negative Path – Unregistered Email
def test_unregistered_email_reset_request(page):
    """Scenario: Requesting a reset for an unregistered email"""
    page.goto(f"{BASE_URL}/forgot-password")
    page.fill("input[name='email']", "unknown@example.com")
    page.click("button:has-text('Send Reset Link')")
    expect(page.locator("text=If an account with that email exists, a reset link has been sent.")).to_be_visible()
    # No email sent – out of scope
    pytest.skip("Email sending verification not implemented")

# Negative Path – Expired Link
def test_expired_reset_link(page):
    """Scenario: Using an expired reset link"""
    # Simulate expired link by navigating to a URL with expired token
    page.goto(f"{BASE_URL}/reset-password?token=expiredtoken")
    expect(page.locator("text=This reset link has expired.")).to_be_visible()
    expect(page.locator("text=Request a new reset link")).to_be_visible()

# Negative Path – Reusing a Link
def test_reusing_reset_link(page):
    """Scenario: Reusing a reset link after it has already been used"""
    page.goto(f"{BASE_URL}/reset-password?token=usedtoken")
    expect(page.locator("text=This reset link has already been used.")).to_be_visible()
    expect(page.locator("text=Request a new reset link")).to_be_visible()

# Negative Path – Password Complexity Failure
def test_weak_password_submission(page):
    """Scenario: Submitting a weak password"""
    page.goto(f"{BASE_URL}/reset-password?token=validtoken")
    page.fill("input[name='newPassword']", "weak")
    page.fill("input[name='confirmPassword']", "weak")
    page.click("button:has-text('Reset Password')")
    expect(page.locator("text=Password must be at least 8 characters and include uppercase, lowercase, number, and symbol.")).to_be_visible()

# Negative Path – Mismatched Passwords
def test_mismatched_passwords(page):
    """Scenario: Submitting mismatched passwords"""
    page.goto(f"{BASE_URL}/reset-password?token=validtoken")
    page.fill("input[name='newPassword']", "StrongPass!23")
    page.fill("input[name='confirmPassword']", "StrongPass!24")
    page.click("button:has-text('Reset Password')")
    expect(page.locator("text=Passwords do not match.")).to_be_visible()

# Edge Case – Multiple Rapid Requests
def test_multiple_rapid_reset_requests(page):
    """Scenario: Requesting multiple reset links in quick succession"""
    page.goto(f"{BASE_URL}/forgot-password")
    page.fill("input[name='email']", "user@example.com")
    for _ in range(3):
        page.click("button:has-text('Send Reset Link')")
    expect(page.locator("text=If an account with that email exists, a reset link has been sent.")).to_be_visible()
    # Throttling and email count verification are out of scope
    pytest.skip("Throttling and email count verification not implemented")

# Edge Case – Email Client Blocks Link
def test_email_client_blocks_link(page):
    """Scenario: Email client blocks the reset link"""
    # Out of scope: cannot simulate email client
    pytest.skip("Email client simulation not implemented")

# Accessibility & Responsiveness
@pytest.mark.parametrize("page_name,device", [
    ("Forgot Password", "desktop"),
    ("Forgot Password", "tablet"),
    ("Forgot Password", "mobile"),
    ("Reset Password", "desktop"),
    ("Reset Password", "tablet"),
    ("Reset Password", "mobile"),
])
def test_accessibility_and_responsiveness(page, page_name, device):
    """Scenario Outline: Page accessibility and responsiveness"""
    # Set viewport based on device
    if device == "desktop":
        page.set_viewport_size({"width": 1280, "height": 800})
    elif device == "tablet":
        page.set_viewport_size({"width": 768, "height": 1024})
    else:
        page.set_viewport_size({"width": 375, "height": 667})
    # Navigate to page
    if page_name == "Forgot Password":
        page.goto(f"{BASE_URL}/forgot-password")
    else:
        page.goto(f"{BASE_URL}/reset-password?token=validtoken")
    # Accessibility check placeholder
    pytest.skip("Accessibility testing not implemented")

# Localization
@pytest.mark.parametrize("lang,language", [
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
])
def test_email_content_localization(page, lang, language):
    """Scenario Outline: Email content localization"""
    # Set language via query param or header
    page.goto(f"{BASE_URL}/forgot-password?lang={lang}")
    page.fill("input[name='email']", f"user_{lang}@example.com")
    page.click("button:has-text('Send Reset Link')")
    expect(page.locator("text=If an account with that email exists, a reset link has been sent.")).to_be_visible()
    # Email content verification out of scope
    pytest.skip("Email content localization verification not implemented")