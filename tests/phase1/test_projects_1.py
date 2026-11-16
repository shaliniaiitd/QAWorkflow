from urllib.parse import urlparse

import pytest
from playwright.sync_api import expect

BASE_URL = "https://shaliniaiitd.github.io"
BOT_BLOCKING_DOMAINS = ("linkedin.com",)


def is_healthy_status(url: str, status: int) -> bool:
    if any(domain in url for domain in BOT_BLOCKING_DOMAINS):
        return status in (200, 999)
    return status == 200


def test_site_loads_and_navigation_works(page):
    """Homepage renders and exposes the real primary nav links."""
    page.goto(BASE_URL)

    brand = page.get_by_role("link", name="Shalini Agarwal home", exact=True)
    expect(brand).to_be_visible()

    nav = page.get_by_role("navigation", name="Primary navigation")
    expect(nav).to_be_visible()

    for link_text in ["Work", "Projects", "Frameworks", "Skills", "Training", "Recognition", "Contact"]:
        link = page.get_by_role("link", name=link_text, exact=True)
        expect(link).to_be_visible()
        expect(link).to_be_enabled()


def test_project_sections_display_complete_information(page):
    """Project sections should have a heading, content, and at least one public link."""
    page.goto(BASE_URL)
    project_articles = page.locator("article").filter(has=page.locator("h3")).filter(has=page.locator("a[href]"))
    assert project_articles.count() > 0, "No project articles found"

    first = project_articles.first
    title = first.locator("h3").first
    description = first.locator("p").first
    link = first.locator("a[href]").first

    expect(title).to_be_visible()
    expect(description).to_be_visible()
    expect(link).to_be_visible()


def test_all_external_urls_reachable(page):
    """External URLs should return healthy status codes."""
    page.goto(BASE_URL)
    links = page.locator('a[href^="http"]')
    hrefs = [link.get_attribute("href") for link in links.all()]
    hrefs = [href for href in hrefs if href and not href.startswith(BASE_URL)]
    assert hrefs, "No external links found on the page"

    for href in hrefs:
        if any(domain in href for domain in BOT_BLOCKING_DOMAINS):
            continue
        response = page.request.get(href)
        assert is_healthy_status(href, response.status), f"{href} returned {response.status}"


def test_images_have_alt_attributes(page):
    """Images should expose an alt attribute even when decorative."""
    page.goto(BASE_URL)
    images = page.locator("img")
    assert images.count() > 0, "No images found"

    for img in images.all():
        alt = img.get_attribute("alt")
        assert alt is not None, f"Image {img.get_attribute('src')} missing alt attribute"


def test_console_is_error_free(page):
    """Console should not emit errors or warnings during page load."""
    page.goto(BASE_URL)
    messages = []

    def handle_console(msg):
        if msg.type in {"error", "warning"}:
            messages.append(msg.text)

    page.on("console", handle_console)
    page.wait_for_timeout(1000)
    assert not messages, f"Console errors/warnings found: {messages}"


def test_mobile_viewport_collapses_navigation(page):
    """Small screens should expose a working toggle menu for the nav."""
    page.set_viewport_size({"width": 320, "height": 480})
    page.goto(BASE_URL)

    hamburger = page.get_by_role("button", name="Toggle navigation", exact=True)
    expect(hamburger).to_be_visible()
    hamburger.click()

    nav = page.locator("nav, header nav").first
    expect(nav).to_be_visible()
    expect(page.locator("a[href='#projects']").first).to_be_visible()


def test_any_linked_project_repository_health(page):
    """Repository links should resolve and not 404."""
    page.goto(BASE_URL)
    github_links = page.locator('a[href*="github.com"]')
    assert github_links.count() > 0, "No GitHub links found"

    href = github_links.first.get_attribute("href")
    assert href is not None and href.startswith("http")
    response = page.request.get(href)
    assert is_healthy_status(href, response.status), f"GitHub link {href} returned {response.status}"
