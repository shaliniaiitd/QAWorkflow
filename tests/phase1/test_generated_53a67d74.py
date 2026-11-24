import pytest
from playwright.sync_api import expect

BASE_URL = "https://shaliniaiitd.github.io"

# Helper functions
def goto_forgot_password(page):
    page.goto(f"{BASE_URL}/forgot-password")

def goto_reset_password(page, token="dummy"):
    page.goto(f"{BASE_URL}/reset-password?token={token}")

@pytest.mark.parametrize("email", ["user@example.com"])
def test_successful_password_reset_request(page, email):
    """Scenario: Successful password reset request"""
    goto_forgot_password(page)
    page.fill("input[name='email']", email)
    page.click("button:has-text('Send Reset Link')")
    expect(page.locator("text=If an account with that email exists, a reset link has been sent.")).to_be_visible()

@pytest.mark.skip(reason="Email token handling not implemented")
def test_resetting_password_with_valid_token(page):
    """Scenario: Resetting the password with a valid token"""
    # Assume we have a valid token from email
    token = "valid-token"
    goto_reset_password(page, token)
    page.fill("input[name='newPassword']", "NewPassw0rd!")
    page.fill("input[name='confirmPassword']", "NewPassw0rd!")
    page.click("button:has-text('Reset Password')")
    expect(page.locator("text=Your password has been successfully reset.")).to_be_visible()
    expect(page).to_have_url(f"{BASE_URL}/login")

@pytest.mark.parametrize("email", ["invalid-email"])
def test_requesting_reset_with_invalid_email_format(page, email):
    """Scenario: Requesting reset with an invalid email format"""
    goto_forgot_password(page)
    page.fill("input[name='email']", email)
    page.click("button:has-text('Send Reset Link')")
    expect(page.locator("text=Please enter a valid email address.")).to_be_visible()

@pytest.mark.skip(reason="Token expiration simulation not available")
def test_using_expired_reset_token(page):
    """Scenario: Using an expired reset token"""
    token = "expired-token"
    goto_reset_password(page, token)
    expect(page.locator("text=Link expired.")).to_be_visible()
    expect(page.locator("text=Resend reset link")).to_be_visible()

@pytest.mark.skip(reason="Password history not accessible in UI")
def test_attempting_to_reuse_last_five_passwords(page):
    """Scenario: Attempting to reuse one of the last five passwords"""
    token = "valid-token"
    goto_reset_password(page, token)
    page.fill("input[name='newPassword']", "OldPassw0rd!")
    page.fill("input[name='confirmPassword']", "OldPassw0rd!")
    page.click("button:has-text('Reset Password')")
    expect(page.locator("text=You cannot reuse any of your last five passwords.")).to_be_visible()

@pytest.mark.parametrize("email", ["ghost@example.com"])
def test_requesting_reset_for_nonexistent_email(page, email):
    """Scenario: Requesting reset for an email that does not exist"""
    goto_forgot_password(page)
    page.fill("input[name='email']", email)
    page.click("button:has-text('Send Reset Link')")
    expect(page.locator("text=If an account with that email exists, a reset link has been sent.")).to_be_visible()

@pytest.mark.skip(reason="Token invalidation logic not testable via UI")
def test_multiple_reset_requests_invalidate_previous_tokens(page):
    """Scenario: Multiple reset requests invalidate previous tokens"""
    email = "user@example.com"
    goto_forgot_password(page)
    page.fill("input[name='email']", email)
    page.click("button:has-text('Send Reset Link')")
    # Assume first token is sent
    page.fill("input[name='email']", email)
    page.click("button:has-text('Send Reset Link')")
    # Assume second token is sent
    # Test that first token is invalid (cannot be used)
    # This would require accessing email inbox, which is out of scope

@pytest.mark.parametrize(
    "page_name,device",
    [
        ("Forgot Password", "Desktop"),
        ("Forgot Password", "Tablet"),
        ("Forgot Password", "Mobile"),
        ("Reset Password", "Desktop"),
        ("Reset Password", "Tablet"),
        ("Reset Password", "Mobile"),
    ],
)
def test_ui_is_wcag_2_1_aa_compliant_and_responsive(page, page_name, device):
    """Scenario Outline: UI is WCAG 2.1 AA compliant and responsive"""
    if page_name == "Forgot Password":
        goto_forgot_password(page)
    else:
        goto_reset_password(page)
    # Resize viewport based on device
    if device == "Desktop":
        page.set_viewport_size({"width": 1920, "height": 1080})
    elif device == "Tablet":
        page.set_viewport_size({"width": 768, "height": 1024})
    else:  # Mobile
        page.set_viewport_size({"width": 375, "height": 667})
    # Check accessible labels
    for element in page.locator("*[role='button'], *[role='link'], input, textarea, select").all():
        aria_label = element.get_attribute("aria-label")
        assert aria_label is not None or element.get_attribute("title") is not None
    # Layout adaptation check (simple example)
    expect(page).to_have_selector("body")

@pytest.mark.parametrize(
    "action,browser,expected",
    [
        ("request reset link", "Chrome", "generic confirmation message"),
        ("request reset link", "Firefox", "generic confirmation message"),
        ("request reset link", "Safari", "generic confirmation message"),
        ("request reset link", "Edge", "generic confirmation message"),
        ("reset password with token", "Chrome", "success message and redirect to login"),
        ("reset password with token", "Firefox", "success message and redirect to login"),
        ("reset password with token", "Safari", "success message and redirect to login"),
        ("reset password with token", "Edge", "success message and redirect to login"),
    ],
)
def test_consistent_behavior_across_browsers(page, action, browser, expected):
    """Scenario Outline: Consistent behavior across browsers"""
    # Note: Browser selection is handled by pytest-playwright config; this test runs in current browser
    if action == "request reset link":
        goto_forgot_password(page)
        page.fill("input[name='email']", "user@example.com")
        page.click("button:has-text('Send Reset Link')")
        expect(page.locator("text=If an account with that email exists, a reset link has been sent.")).to_be_visible()
    else:
        token = "valid-token"
        goto_reset_password(page, token)
        page.fill("input[name='newPassword']", "NewPassw0rd!")
        page.fill("input[name='confirmPassword']", "NewPassw0rd!")
        page.click("button:has-text('Reset Password')")
        expect(page.locator("text=Your password has been successfully reset.")).to_be_visible()
        expect(page).to_have_url(f"{BASE_URL}/login")
    # Verify expected outcome
    if expected == "generic confirmation message":
        expect(page.locator("text=If an account with that email exists, a reset link has been sent.")).to_be_visible()
    else:
        expect(page.locator("text=Your password has been successfully reset.")).to_be_visible()
        expect(page).to_have_url(f"{BASE_URL}/login")