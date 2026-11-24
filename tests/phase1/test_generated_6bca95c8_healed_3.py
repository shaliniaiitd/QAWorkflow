import pytest
from playwright.sync_api import expect

@pytest.fixture(scope="session")
def base_url():
    return "https://shaliniaiitd.github.io"

def test_successful_password_reset(page, base_url):
    """Scenario: Successful password reset"""
    page.goto(f"{base_url}/forgot-password")
    page.wait_for_selector('input[name="email"]', timeout=60000)
    page.fill('input[name="email"]', "alice@example.com")
    page.click('button[type="submit"]')
    expect(page.locator('text=If this email is registered, a reset link has been sent')).to_be_visible()
    reset_link = page.locator('a:has-text("Reset password")')
    reset_link.click()
    expect(page).to_have_url(f"{base_url}/reset-password")
    expect(page).to_have_title("Reset password")
    assert page.url.startswith("https://")
    expect(page.locator('input[name="newPassword"]')).to_be_visible()
    expect(page.locator('input[name="confirmPassword"]')).to_be_visible()
    expect(page.locator('button[type="submit"]').first).to_be_visible()
    page.wait_for_selector('input[name="newPassword"]', timeout=60000)
    page.fill('input[name="newPassword"]', "NewPassw0rd!")
    page.wait_for_selector('input[name="confirmPassword"]', timeout=60000)
    page.fill('input[name="confirmPassword"]', "NewPassw0rd!")
    page.click('button[type="submit"]')
    expect(page.locator('text=Your password has been reset')).to_be_visible()
    page.goto(f"{base_url}/login")
    page.wait_for_selector('input[name="email"]', timeout=60000)
    page.fill('input[name="email"]', "alice@example.com")
    page.wait_for_selector('input[name="password"]', timeout=60000)
    page.fill('input[name="password"]', "NewPassw0rd!")
    page.click('button[type="submit"]')
    expect(page.locator('text=Welcome, alice@example.com')).to_be_visible()

def test_unregistered_email_reset(page, base_url):
    """Scenario: Request reset with an unregistered email"""
    page.goto(f"{base_url}/forgot-password")
    page.wait_for_selector('input[name="email"]', timeout=60000)
    page.fill('input[name="email"]', "unknown@example.com")
    page.click('button[type="submit"]')
    expect(page.locator('text=If this email is registered, a reset link has been sent')).to_be_visible()

def test_token_reuse(page, base_url):
    """Scenario: Attempt to reuse a reset token"""
    page.goto(f"{base_url}/reset-password?token=validtoken")
    page.wait_for_selector('input[name="newPassword"]', timeout=60000)
    page.fill('input[name="newPassword"]', "ValidPass1!")
    page.wait_for_selector('input[name="confirmPassword"]', timeout=60000)
    page.fill('input[name="confirmPassword"]', "ValidPass1!")
    page.click('button[type="submit"]')
    expect(page.locator('text=Your password has been reset')).to_be_visible()
    page.goto(f"{base_url}/reset-password?token=validtoken")
    expect(page.locator('text=Invalid or expired token')).to_be_visible()

def test_expired_token(page, base_url):
    """Scenario: Use an expired reset token"""
    page.goto(f"{base_url}/reset-password?token=expiredtoken")
    expect(page.locator('text=Invalid or expired token')).to_be_visible()

def test_password_complexity_failure(page, base_url):
    """Scenario: New password does not meet complexity requirements"""
    page.goto(f"{base_url}/reset-password?token=validtoken")
    page.wait_for_selector('input[name="newPassword"]', timeout=60000)
    page.fill('input[name="newPassword"]', "short")
    page.wait_for_selector('input[name="confirmPassword"]', timeout=60000)
    page.fill('input[name="confirmPassword"]', "short")
    page.click('button[type="submit"]')
    expect(page.locator('text=Password does not meet complexity requirements')).to_be_visible()

def test_mismatched_confirmation(page, base_url):
    """Scenario: Password and confirmation do not match"""
    page.goto(f"{base_url}/reset-password?token=validtoken")
    page.wait_for_selector('input[name="newPassword"]', timeout=60000)
    page.fill('input[name="newPassword"]', "ValidPass1!")
    page.wait_for_selector('input[name="confirmPassword"]', timeout=60000)
    page.fill('input[name="confirmPassword"]', "DifferentPass1!")
    page.click('button[type="submit"]')
    expect(page.locator('text=Passwords do not match')).to_be_visible()

def test_tampered_token(page, base_url):
    """Scenario: Tampered or malformed token"""
    page.goto(f"{base_url}/reset-password?token=abc123")
    page.goto(f"{base_url}/reset-password?token=xyz789")
    expect(page.locator('text=Invalid or expired token')).to_be_visible()

def test_smtp_failure(page, base_url):
    """Scenario: SMTP failure when sending reset email"""
    page.goto(f"{base_url}/forgot-password")
    page.wait_for_selector('input[name="email"]', timeout=60000)
    page.fill('input[name="email"]', "smtpdown@example.com")
    page.click('button[type="submit"]')
    expect(page.locator('text=Unable to send email. Please try again later.')).to_be_visible()
    expect(page.locator('button:has-text("Retry")')).to_be_visible()

def test_logged_in_user_requests_reset(page, base_url):
    """Scenario: User already logged in requests a reset"""
    page.goto(f"{base_url}/login")
    page.wait_for_selector('input[name="email"]', timeout=60000)
    page.fill('input[name="email"]', "alice@example.com")
    page.wait_for_selector('input[name="password"]', timeout=60000)
    page.fill('input[name="password"]', "OldPassw0rd!")
    page.click('button[type="submit"]')
    expect(page.locator('text=Welcome, alice@example.com')).to_be_visible()
    page.goto(f"{base_url}/forgot-password")
    page.click('button[type="submit"]')
    expect(page.locator('text=Login')).to_be_visible()
    expect(page.locator('text=If this email is registered, a reset link has been sent')).to_be_visible()

def test_browser_back_after_reset(page, base_url):
    """Scenario: Back button after successful reset"""
    page.goto(f"{base_url}/forgot-password")
    page.wait_for_selector('input[name="email"]', timeout=60000)
    page.fill('input[name="email"]', "alice@example.com")
    page.click('button[type="submit"]')
    page.goto(f"{base_url}/reset-password?token=validtoken")
    page.wait_for_selector('input[name="newPassword"]', timeout=60000)
    page.fill('input[name="newPassword"]', "NewPassw0rd!")
    page.wait_for_selector('input[name="confirmPassword"]', timeout=60000)
    page.fill('input[name="confirmPassword"]', "NewPassw0rd!")
    page.click('button[type="submit"]')
    expect(page.locator('text=Your password has been reset')).to_be_visible()
    page.go_back()
    expect(page).to_have_url(f"{base_url}/login")
    page.goto(f"{base_url}/reset-password?token=validtoken")
    expect(page.locator('text=Invalid or expired token')).to_be_visible()

def test_accessibility_compliance(page, base_url):
    """Scenario: Reset page is WCAG 2.1 AA compliant"""
    page.goto(f"{base_url}/reset-password?token=validtoken")
    expect(page.locator('label[for="newPassword"]')).to_have_text("New Password")
    expect(page.locator('label[for="confirmPassword"]')).to_have_text("Confirm Password")

def test_rate_limiting(page, base_url):
    """Scenario: Exceeding rate limit for reset requests"""
    page.goto(f"{base_url}/forgot-password")
    for _ in range(5):
        page.wait_for_selector('input[name="email"]', timeout=60000)
        page.fill('input[name="email"]', "alice@example.com")
        page.click('button[type="submit"]')
        expect(page.locator('text=If this email is registered, a reset link has been sent')).to_be_visible()
    page.wait_for_selector('input[name="email"]', timeout=60000)
    page.fill('input[name="email"]', "alice@example.com")
    page.click('button[type="submit"]')
    expect(page.locator('text=Too many requests. Please try again later.')).to_be_visible()

@pytest.mark.parametrize("locale", ["en-US", "es-ES", "fr-FR"])
def test_localization(page, base_url, locale):
    """Scenario Outline: Reset flow in supported locales"""
    page.goto(f"{base_url}/forgot-password")
    page.select_option('select[name="language"]', locale)
    page.wait_for_selector('input[name="email"]', timeout=60000)
    page.fill('input[name="email"]', "alice@example.com")
    page.click('button[type="submit"]')
    if locale == "en-US":
        expect(page.locator('text=If this email is registered, a reset link has been sent')).to_be_visible()
    elif locale == "es-ES":
        expect(page.locator('text=Si este correo está registrado, se ha enviado un enlace de restablecimiento')).to_be_visible()
    elif locale == "fr-FR":
        expect(page.locator('text=Si cet e‑mail est enregistré, un lien de réinitialisation a été envoyé')).to_be_visible()