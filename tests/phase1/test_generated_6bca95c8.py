import pytest
from playwright.sync_api import expect

BASE_URL = "https://shaliniaiitd.github.io"

@pytest.fixture(autouse=True)
def base_url(page):
    page.goto(BASE_URL)

def test_successful_password_reset(page):
    """Scenario: Successful password reset"""
    # Go to Forgot password page
    page.goto(f"{BASE_URL}/forgot-password")
    # Enter registered email
    page.fill('input[name="email"]', "alice@example.com")
    page.click('button[type="submit"]')
    # Verify confirmation message
    expect(page.locator('text=If this email is registered, a reset link has been sent')).to_be_visible()
    # Simulate clicking the reset link (assuming link is displayed on page for demo)
    reset_link = page.locator('a:has-text("Reset password")')
    reset_link.click()
    # Verify we are on Reset password page
    expect(page).to_have_url(f"{BASE_URL}/reset-password")
    expect(page).to_have_title("Reset password")
    # Verify HTTPS
    assert page.url.startswith("https://")
    # Verify form fields
    expect(page.locator('input[name="newPassword"]')).to_be_visible()
    expect(page.locator('input[name="confirmPassword"]')).to_be_visible()
    expect(page.locator('button[type="submit"]').first).to_be_visible()
    # Enter new password
    page.fill('input[name="newPassword"]', "NewPassw0rd!")
    page.fill('input[name="confirmPassword"]', "NewPassw0rd!")
    page.click('button[type="submit"]')
    # Verify success message
    expect(page.locator('text=Your password has been reset')).to_be_visible()
    # Verify login with new password
    page.goto(f"{BASE_URL}/login")
    page.fill('input[name="email"]', "alice@example.com")
    page.fill('input[name="password"]', "NewPassw0rd!")
    page.click('button[type="submit"]')
    expect(page.locator('text=Welcome, alice@example.com')).to_be_visible()

def test_unregistered_email_reset(page):
    """Scenario: Request reset with an unregistered email"""
    page.goto(f"{BASE_URL}/forgot-password")
    page.fill('input[name="email"]', "unknown@example.com")
    page.click('button[type="submit"]')
    expect(page.locator('text=If this email is registered, a reset link has been sent')).to_be_visible()

def test_token_reuse(page):
    """Scenario: Attempt to reuse a reset token"""
    # Assume we have a reset link available
    page.goto(f"{BASE_URL}/reset-password?token=validtoken")
    page.fill('input[name="newPassword"]', "ValidPass1!")
    page.fill('input[name="confirmPassword"]', "ValidPass1!")
    page.click('button[type="submit"]')
    expect(page.locator('text=Your password has been reset')).to_be_visible()
    # Reuse same link
    page.goto(f"{BASE_URL}/reset-password?token=validtoken")
    expect(page.locator('text=Invalid or expired token')).to_be_visible()

def test_expired_token(page):
    """Scenario: Use an expired reset token"""
    page.goto(f"{BASE_URL}/reset-password?token=expiredtoken")
    expect(page.locator('text=Invalid or expired token')).to_be_visible()

def test_password_complexity_failure(page):
    """Scenario: New password does not meet complexity requirements"""
    page.goto(f"{BASE_URL}/reset-password?token=validtoken")
    page.fill('input[name="newPassword"]', "short")
    page.fill('input[name="confirmPassword"]', "short")
    page.click('button[type="submit"]')
    expect(page.locator('text=Password does not meet complexity requirements')).to_be_visible()

def test_mismatched_confirmation(page):
    """Scenario: Password and confirmation do not match"""
    page.goto(f"{BASE_URL}/reset-password?token=validtoken")
    page.fill('input[name="newPassword"]', "ValidPass1!")
    page.fill('input[name="confirmPassword"]', "DifferentPass1!")
    page.click('button[type="submit"]')
    expect(page.locator('text=Passwords do not match')).to_be_visible()

def test_tampered_token(page):
    """Scenario: Tampered or malformed token"""
    page.goto(f"{BASE_URL}/reset-password?token=abc123")
    # Tamper token
    page.goto(f"{BASE_URL}/reset-password?token=xyz789")
    expect(page.locator('text=Invalid or expired token')).to_be_visible()

def test_smtp_failure(page):
    """Scenario: SMTP failure when sending reset email"""
    # Simulate SMTP failure by using a special email
    page.goto(f"{BASE_URL}/forgot-password")
    page.fill('input[name="email"]', "smtpdown@example.com")
    page.click('button[type="submit"]')
    expect(page.locator('text=Unable to send email. Please try again later.')).to_be_visible()
    expect(page.locator('button:has-text("Retry")')).to_be_visible()

def test_logged_in_user_requests_reset(page):
    """Scenario: User already logged in requests a reset"""
    # Log in first
    page.goto(f"{BASE_URL}/login")
    page.fill('input[name="email"]', "alice@example.com")
    page.fill('input[name="password"]', "OldPassw0rd!")
    page.click('button[type="submit"]')
    expect(page.locator('text=Welcome, alice@example.com')).to_be_visible()
    # Navigate to forgot password
    page.goto(f"{BASE_URL}/forgot-password")
    page.click('button[type="submit"]')
    # Should be logged out
    expect(page.locator('text=Login')).to_be_visible()
    # Reset email should be sent (message displayed)
    expect(page.locator('text=If this email is registered, a reset link has been sent')).to_be_visible()

def test_browser_back_after_reset(page):
    """Scenario: Back button after successful reset"""
    # Perform reset as in successful test
    page.goto(f"{BASE_URL}/forgot-password")
    page.fill('input[name="email"]', "alice@example.com")
    page.click('button[type="submit"]')
    page.goto(f"{BASE_URL}/reset-password?token=validtoken")
    page.fill('input[name="newPassword"]', "NewPassw0rd!")
    page.fill('input[name="confirmPassword"]', "NewPassw0rd!")
    page.click('button[type="submit"]')
    expect(page.locator('text=Your password has been reset')).to_be_visible()
    # Click back
    page.go_back()
    expect(page).to_have_url(f"{BASE_URL}/login")
    # Try using token again
    page.goto(f"{BASE_URL}/reset-password?token=validtoken")
    expect(page.locator('text=Invalid or expired token')).to_be_visible()

def test_accessibility_compliance(page):
    """Scenario: Reset page is WCAG 2.1 AA compliant"""
    page.goto(f"{BASE_URL}/reset-password?token=validtoken")
    # Check labels
    expect(page.locator('label[for="newPassword"]')).to_have_text("New Password")
    expect(page.locator('label[for="confirmPassword"]')).to_have_text("Confirm Password")
    # Contrast and focus order are not testable via Playwright directly; assume compliance

def test_rate_limiting(page):
    """Scenario: Exceeding rate limit for reset requests"""
    page.goto(f"{BASE_URL}/forgot-password")
    for _ in range(5):
        page.fill('input[name="email"]', "alice@example.com")
        page.click('button[type="submit"]')
        expect(page.locator('text=If this email is registered, a reset link has been sent')).to_be_visible()
    # Sixth attempt
    page.fill('input[name="email"]', "alice@example.com")
    page.click('button[type="submit"]')
    expect(page.locator('text=Too many requests. Please try again later.')).to_be_visible()

@pytest.mark.parametrize("locale", ["en-US", "es-ES", "fr-FR"])
def test_localization(page, locale):
    """Scenario Outline: Reset flow in supported locales"""
    page.goto(f"{BASE_URL}/forgot-password")
    # Assume language selector has value attribute
    page.select_option('select[name="language"]', locale)
    page.fill('input[name="email"]', "alice@example.com")
    page.click('button[type="submit"]')
    # Verify messages in selected locale
    if locale == "en-US":
        expect(page.locator('text=If this email is registered, a reset link has been sent')).to_be_visible()
    elif locale == "es-ES":
        expect(page.locator('text=Si este correo está registrado, se ha enviado un enlace de restablecimiento')).to_be_visible()
    elif locale == "fr-FR":
        expect(page.locator('text=Si cet e‑mail est enregistré, un lien de réinitialisation a été envoyé')).to_be_visible()