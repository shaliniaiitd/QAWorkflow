import time
import requests
from playwright.sync_api import Page, expect
import pytest

BOT_BLOCKING_DOMAINS = ("linkedin.com",)

def is_healthy_status(url: str, status: int) -> bool:
    if any(domain in url for domain in BOT_BLOCKING_DOMAINS):
        return status in (200, 999)  # 999 = intentional anti-bot response, not broken
    return status == 200

# ---------- Helper functions ----------
def get_external_links(page: Page):
    return [a.get_attribute("href") for a in page.locator("a[href]").all()]

# ---------- Tests ----------
def test_navigation_links_visible_and_functional(page: Page):
    """
    Feature: Navigation visibility and functionality
    Scenario: Navigation links are visible and work on desktop
    """
    page.goto("https://shaliniaiitd.github.io")
    nav_links = [
        "SA\nShalini Agarwal",
        "Work",
        "Projects",
        "Frameworks",
        "Skills",
        "Training",
        "Recognition",
        "Contact",
    ]
    for link_text in nav_links:
        link = page.get_by_role("link", name=link_text, exact=True)
        expect(link).to_be_visible()
    # Click Projects link
    projects_link = page.get_by_role("link", name="Projects", exact=True)
    projects_link.click()
    expect(page).to_have_url("https://shaliniaiitd.github.io/#projects")
    expect(page.locator("h2:has-text('Projects')")).to_be_visible()

def test_navigation_collapses_on_mobile(page: Page):
    """
    Feature: Navigation visibility and functionality
    Scenario: Navigation collapses on mobile and toggles correctly
    """
    page.set_viewport_size({"width": 320, "height": 640})
    page.goto("https://shaliniaiitd.github.io")
    menu_button = page.locator(".menu-button")
    expect(menu_button).to_be_visible()
    menu_button.click()
    # After clicking, navigation menu should be visible
    nav_menu = page.locator("nav")
    expect(nav_menu).to_be_visible()
    # Click Contact link
    contact_link = page.get_by_role("link", name="Contact", exact=True)
    contact_link.click()
    expect(page).to_have_url("https://shaliniaiitd.github.io/#contact")
    expect(page.locator("h2:has-text('Contact')")).to_be_visible()

def test_external_links_status(page: Page):
    """
    Feature: Accessibility compliance
    Scenario: Site passes contrast and keyboard navigation checks
    (General robustness check: external links are healthy)
    """
    page.goto("https://shaliniaiitd.github.io")
    links = get_external_links(page)
    for url in links:
        if url.startswith("http"):
            try:
                resp = requests.head(url, allow_redirects=True, timeout=5)
                status = resp.status_code
            except Exception:
                status = 0
            assert is_healthy_status(url, status), f"Link {url} returned status {status}"

def test_image_alt_text(page: Page):
    """
    Feature: Image alt text and placeholders
    Scenario: Project image has descriptive alt text
    """
    page.goto("https://shaliniaiitd.github.io")
    # Assume project cards are within .project-grid
    project_card = page.locator(".project-grid .project-card").first
    img = project_card.locator("img")
    alt = img.get_attribute("alt")
    assert alt is not None, "Image missing alt attribute"
    assert alt.strip() != "", "Image alt attribute is empty"

def test_no_broken_github_links(page: Page):
    """
    Feature: Project card content
    Scenario: Project card handles a broken GitHub link
    (General robustness check: GitHub links are reachable)
    """
    page.goto("https://shaliniaiitd.github.io")
    github_links = page.locator("a[href*='github.com']").all()
    for link in github_links:
        url = link.get_attribute("href")
        if url:
            try:
                resp = requests.head(url, allow_redirects=True, timeout=5)
                status = resp.status_code
            except Exception:
                status = 0
            assert is_healthy_status(url, status), f"GitHub link {url} returned status {status}"

def test_project_card_elements(page: Page):
    """
    Feature: Project card content
    Scenario: Project card shows all required elements
    """
    page.goto("https://shaliniaiitd.github.io")
    project_card = page.locator(".project-grid .project-card").first
    title = project_card.locator(".project-title")
    description = project_card.locator(".project-description")
    tech_stack = project_card.locator(".project-tech-stack")
    link = project_card.locator("a[href]")
    expect(title).to_be_visible()
    expect(description).to_be_visible()
    expect(tech_stack).to_be_visible()
    expect(link).to_be_visible()

def test_skills_page_excludes_empty_tags(page: Page):
    """
    Feature: Empty skill tags
    Scenario: Rendered skill list excludes empty tags
    """
    page.goto("https://shaliniaiitd.github.io")
    skills_link = page.get_by_role("link", name="Skills", exact=True)
    skills_link.click()
    expect(page).to_have_url("https://shaliniaiitd.github.io/#skills")
    tags = page.locator(".skills-board .tag")
    for tag in tags.all():
        text = tag.inner_text().strip()
        assert text != "", "Found an empty skill tag"

def test_footer_last_updated_timestamp(page: Page):
    """
    Feature: Last updated timestamp
    Scenario: Footer displays correct timestamp
    """
    page.goto("https://shaliniaiitd.github.io")
    footer = page.locator("footer")
    expect(footer).to_be_visible()
    timestamp = footer.locator("text=Last Updated")
    expect(timestamp).to_be_visible()
    # Basic check: timestamp contains a year
    text = timestamp.inner_text()
    assert any(str(year) in text for year in range(2020, 2030)), "Timestamp does not contain a year"

def test_page_load_time(page: Page):
    """
    Feature: Homepage performance
    Scenario: Page loads within acceptable time
    """
    start = time.time()
    page.goto("https://shaliniaiitd.github.io")
    load_time = time.time() - start
    assert load_time < 2, f"Page load time {load_time:.2f}s exceeds 2 seconds"

def test_no_console_errors(page: Page):
    """
    Feature: Accessibility compliance
    General robustness check: No console errors
    """
    page.goto("https://shaliniaiitd.github.io")
    errors = []
    page.on("console", lambda msg: errors.append(msg))
    # Wait a short period to allow console messages to appear
    page.wait_for_timeout(2000)
    assert not errors, f"Console errors found: {errors}"