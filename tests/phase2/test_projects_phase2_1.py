import time
from urllib.parse import urlparse

import pytest
from playwright.sync_api import expect, Page

BOT_BLOCKING_DOMAINS = ("linkedin.com",)


def is_healthy_status(url: str, status: int) -> bool:
    if any(domain in url for domain in BOT_BLOCKING_DOMAINS):
        return status in (200, 999)  # 999 = intentional anti-bot response, not broken
    return status == 200


NAV_LINKS = [
    "Work",
    "Projects",
    "Frameworks",
    "Skills",
    "Training",
    "Recognition",
    "Contact",
]


@pytest.fixture
def open_homepage(page: Page):
    """Open the portfolio site and return the page."""
    page.goto("https://shaliniaiitd.github.io")
    return page


def test_homepage_load_time_and_navigation(open_homepage: Page):
    """
    Scenario: Homepage loads quickly and the header interaction is available.
    """
    page = open_homepage
    start = time.time()
    page.wait_for_load_state("networkidle")
    load_time = time.time() - start
    assert load_time < 2, f"Homepage load time {load_time:.2f}s exceeded 2s"

    brand_link = page.get_by_role("link", name="Shalini Agarwal home", exact=True)
    expect(brand_link).to_be_visible()
    expect(brand_link).to_be_enabled()

    nav = page.get_by_role("navigation", name="Primary navigation")
    expect(nav).to_be_visible()

    for link_text in NAV_LINKS:
        link = page.get_by_role("link", name=link_text, exact=True)
        expect(link).to_be_visible()
        expect(link).to_be_enabled()


def test_project_cards_and_external_links(open_homepage: Page):
    """
    Scenario: Project sections show meaningful content and their public links resolve.
    """
    page = open_homepage

    project_cards = page.locator("article").filter(has=page.locator("h3")).filter(has=page.locator("a[href]"))
    assert project_cards.count() > 0, "No project articles found"

    first_card = project_cards.first
    title = first_card.locator("h3").first
    description = first_card.locator("p").nth(0)
    link = first_card.locator("a[href]").first

    expect(title).to_be_visible()
    expect(description).to_be_visible()
    expect(link).to_be_visible()

    href = link.get_attribute("href")
    assert href is not None, "Project card link missing href"

    if href.startswith("http"):
        response = page.request.get(href)
        assert is_healthy_status(href, response.status), f"Link {href} returned {response.status}"


def test_images_have_alt_attributes(open_homepage: Page):
    """
    Scenario: Accessibility and image alt text compliance
    """
    page = open_homepage
    images = page.locator("img")
    assert images.count() > 0, "No images found on the page"

    for img in images.all():
        alt = img.get_attribute("alt")
        assert alt is not None, f"Image {img.get_attribute('src')} missing alt attribute"


def test_mobile_navigation_collapses(open_homepage: Page):
    """
    Scenario: Mobile navigation exposes a working toggle menu on small screens.
    """
    page = open_homepage
    page.set_viewport_size({"width": 320, "height": 480})
    page.wait_for_load_state("networkidle")

    menu_button = page.get_by_role("button", name="Toggle navigation", exact=True)
    expect(menu_button).to_be_visible()
    menu_button.click()

    nav = page.locator("nav, header nav").first
    expect(nav).to_be_visible()
    expect(page.locator("a[href='#projects']").first).to_be_visible()


def test_external_links_status_codes(open_homepage: Page):
    """
    General robustness check: all external links return healthy status codes
    """
    page = open_homepage
    external_links = page.locator("a[href^='http']")
    count = external_links.count()
    assert count > 0, "No external links found"

    for link in external_links.all():
        href = link.get_attribute("href")
        if href:
            response = page.request.get(href)
            assert is_healthy_status(href, response.status), f"Link {href} returned {response.status}"
        else:
            pytest.fail("External link without href attribute found")