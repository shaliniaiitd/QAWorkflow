import time
import requests
from playwright.sync_api import Page, expect
import pytest

BOT_BLOCKING_DOMAINS = ("linkedin.com",)

def is_healthy_status(url: str, status: int) -> bool:
    if any(domain in url for domain in BOT_BLOCKING_DOMAINS):
        return status in (200, 999)  # 999 = intentional anti-bot response, not broken
    return status == 200

NAV_LINKS = ["Work", "Projects", "Frameworks", "Skills", "Training", "Recognition", "Contact"]
PROJECT_HEADINGS = [
    "Schema-Agnostic API and Data Validation",
    "RMBS Multi-Service QA",
    "Big Data Validation and LLM Evaluation",
    "BDD Regression Framework",
    "WealthDesk Agent",
    "AgenticQAWorkflows",
    "Training Portal Suite",
    "Python Self-Learning",
    "Epex Spot Test Suite",
    "Playwright JS POM Framework",
    "Playwright Assignments",
]

@pytest.fixture(scope="session")
def base_url():
    return "https://shaliniaiitd.github.io"

def test_happy_path_core_content_loads_quickly_and_is_accessible(page: Page, base_url: str):
    """
    Scenario: Happy path – All core content loads quickly and is accessible
    """
    start = time.perf_counter()
    page.goto(base_url, wait_until="load")
    elapsed = time.perf_counter() - start
    assert elapsed < 2, f"Page load time {elapsed:.2f}s exceeds 2s"

    # Brand link
    brand = page.get_by_role("link", name="Shalini Agarwal home", exact=True)
    expect(brand).to_be_visible()
    expect(brand).to_be_enabled()

    # Navigation links
    for link_text in NAV_LINKS:
        nav_link = page.get_by_role("link", name=link_text, exact=True)
        expect(nav_link).to_be_visible()
        expect(nav_link).to_be_enabled()

    # Projects section
    projects_section = page.locator("#projects")
    expect(projects_section).to_be_visible()
    articles = projects_section.locator("article")
    assert articles.count() > 0, "No project articles found"

    for heading in PROJECT_HEADINGS:
        article = projects_section.locator(f"article:has(h2:has-text('{heading}'))")
        expect(article).to_have_count(1)
        # Check for a GitHub link inside the article
        github_link = article.locator("a[href*='github.com']")
        expect(github_link).to_have_count(1)
        expect(github_link).to_be_visible()
        expect(github_link).to_be_enabled()

    # External links status
    external_links = page.locator("a[href^='http']")
    hrefs = [el.get_attribute("href") for el in external_links.all()]
    for href in hrefs:
        if href is None:
            continue
        try:
            resp = requests.head(href, allow_redirects=True, timeout=5)
            status = resp.status_code
        except Exception:
            status = 0
        assert is_healthy_status(href, status), f"External link {href} returned status {status}"

    # Images alt attribute
    images = page.locator("img")
    for img in images.all():
        alt = img.get_attribute("alt")
        assert alt is not None, f"Image {img.get_attribute('src')} missing alt attribute"

def test_negative_path_unsupported_browser_and_missing_project_link(page: Page, base_url: str):
    """
    Scenario: Negative path – Unsupported browser and missing project link
    """
    # Playwright cannot emulate IE11; we skip the browser check but ensure page loads
    page.goto(base_url, wait_until="load")
    # Graceful fallback message for unsupported browsers is not present; we check for its absence
    fallback = page.locator("text=Your browser is not supported")
    assert fallback.count() == 0, "Fallback message displayed for supported browser"

    # Core content should still be rendered
    for link_text in NAV_LINKS:
        nav_link = page.get_by_role("link", name=link_text, exact=True)
        expect(nav_link).to_be_visible()

    # Projects section
    projects_section = page.locator("#projects")
    expect(projects_section).to_be_visible()
    articles = projects_section.locator("article")
    assert articles.count() > 0, "No project articles found"

    # Check that any private GitHub repo link is hidden or replaced
    private_links = articles.locator("a[href*='github.com'][data-private='true']")
    assert private_links.count() == 0, "Private GitHub links are visible"

    # Footer sticky on small screen
    page.set_viewport_size({"width": 320, "height": 480})
    footer = page.locator("footer")
    expect(footer).to_be_visible()
    # Verify footer is sticky by checking its position after scrolling
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    footer_y = footer.bounding_box()["y"]
    page.evaluate("window.scrollTo(0, 0)")
    top_footer_y = footer.bounding_box()["y"]
    assert top_footer_y == footer_y, "Footer is not sticky on small screen"