import pytest
from playwright.sync_api import expect

BASE_URL = "https://shaliniaiitd.github.io"

@pytest.fixture(autouse=True)
def base_page(page):
    """Ensure each test starts from the base URL."""
    page.goto(BASE_URL)

def test_successful_password_reset_flow(page):
    """Scenario: Successful password reset flow"""
    # Go to Forgot Password page
    page.goto(f"{BASE_URL}/forgot-password")
    # Enter valid email
    page.fill("input[name='email']", "user@example.com")
    # Click Send reset link
    page.click("button:has-text('Send reset link')")
    # Verify confirmation message
    expect(page.locator("text=If an account with that email exists, a reset link has been sent.")).to_be_visible()
    # Simulate clicking the link in the email (placeholder token)
    page.goto(f"{BASE_URL}/reset-password?token=valid-token")
    # Verify Reset Password form
    expect(page.locator("h1:has-text('Reset Password')")).to_be_visible()
    expect(page.locator("input[name='newPassword']")).to_be_visible()
    expect(page.locator("input[name='confirmPassword']")).to_be_visible()
    # Enter strong password
    page.fill("input[name='newPassword']", "StrongPass!23")
    page.fill("input[name='confirmPassword']", "StrongPass!23")
    # Submit form
    page.click("button:has-text('Reset Password')")
    # Verify success message
    expect(page.locator("text=Your password has been reset successfully.")).to_be_visible()
    # Verify login with new password (placeholder)
    page.goto(f"{BASE_URL}/login")
    page.fill("input[name='email']", "user@example.com")
    page.fill("input[name='password']", "StrongPass!23")
    page.click("button:has-text('Login')")
    expect(page.locator("text=Welcome, user@example.com")).to_be_visible()

def test_expired_reset_link(page):
    """Scenario: Attempt to use an expired reset link"""
    page.goto(f"{BASE_URL}/reset-password?token=expired-token")
    expect(page.locator("text=The reset link has expired.")).to_be_visible()
    expect(page.locator("text=Please request a new reset link.")).to_be_visible()

def test_reused_reset_link(page):
    """Scenario: Attempt to reuse a reset link"""
    page.goto(f"{BASE_URL}/reset-password?token=used-token")
    expect(page.locator("text=This reset link has already been used.")).to_be_visible()
    expect(page.locator("text=Please request a new reset link.")).to_be_visible()

def test_deleted_account_reset_link(page):
    """Scenario: Reset link for a deleted account"""
    page.goto(f"{BASE_URL}/reset-password?token=deleted-token")
    expect(page.locator("text=Account not found.")).to_be_visible()
    expect(page.locator("text=Please create a new account.")).to_be_visible()

def test_invalid_email_request(page):
    """Scenario: Request reset for a non‑existent email"""
    page.goto(f"{BASE_URL}/forgot-password")
    page.fill("input[name='email']", "unknown@example.com")
    page.click("button:has-text('Send reset link')")
    expect(page.locator("text=If an account with that email exists, a reset link has been sent.")).to_be_visible()
    # No email sent – cannot verify in UI; assume success

def test_weak_password(page):
    """Scenario: Enter a password that fails complexity"""
    page.goto(f"{BASE_URL}/reset-password?token=valid-token")
    page.fill("input[name='newPassword']", "weak")
    page.fill("input[name='confirmPassword']", "weak")
    page.click("button:has-text('Reset Password')")
    expect(page.locator("text=Password does not meet complexity requirements.")).to_be_visible()

def test_mismatched_confirmation(page):
    """Scenario: Password confirmation does not match"""
    page.goto(f"{BASE_URL}/reset-password?token=valid-token")
    page.fill("input[name='newPassword']", "StrongPass!23")
    page.fill("input[name='confirmPassword']", "StrongPass!24")
    page.click("button:has-text('Reset Password')")
    expect(page.locator("text=Passwords do not match.")).to_be_visible()

def test_already_logged_in_reset_request(page):
    """Scenario: User requests reset while already logged in"""
    # Simulate login
    page.goto(f"{BASE_URL}/login")
    page.fill("input[name='email']", "user@example.com")
    page.fill("input[name='password']", "StrongPass!23")
    page.click("button:has-text('Login')")
    expect(page.locator("text=Welcome, user@example.com")).to_be_visible()
    # Navigate to Forgot Password
    page.goto(f"{BASE_URL}/forgot-password")
    expect(page.locator("text=You are already logged in. To reset your password, please log out first.")).to_be_visible()
    expect(page.locator("h1:has-text('Account Settings')")).to_be_visible()

def test_rate_limiting_reset_requests(page):
    """Scenario: Rapidly request multiple reset emails"""
    page.goto(f"{BASE_URL}/forgot-password")
    page.fill("input[name='email']", "user@example.com")
    for _ in range(3):
        page.click("button:has-text('Send reset link')")
    expect(page.locator("text=Please wait before requesting another reset link.")).to_be_visible()
    # Assume only one email sent – cannot verify in UI

def test_smtp_failure_on_reset_email(page):
    """Scenario: SMTP failure when sending reset email"""
    page.goto(f"{BASE_URL}/forgot-password")
    page.fill("input[name='email']", "user@example.com")
    page.click("button:has-text('Send reset link')")
    # Simulate SMTP failure by expecting error message
    expect(page.locator("text=Unable to send email. Please try again.")).to_be_visible()
    expect(page.locator("button:has-text('Retry')")).to_be_visible()
    # Click Retry
    page.click("button:has-text('Retry')")
    # After retry, assume success
    expect(page.locator("text=If an account with that email exists, a reset link has been sent.")).to_be_visible()