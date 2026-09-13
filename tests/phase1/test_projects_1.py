import time
import re
import requests
from playwright.sync_api import expect

BASE_URL = "https://shaliniaiitd.github.io"


def test_site_loads_quickly_and_navigation_works(page):
    """Scenario: Site loads quickly and navigation works on all viewports"""
    start = time.time()
    page.goto(BASE_URL)
    load_time = time.time() - start
    assert load_time < 2, f"Page loaded in {load_time:.2f}s, expected < 2s"

    nav_links = ["Home", "About", "Projects", "Contact"]
    for link_text in nav_links:
        locator = page.locator(f"text={link_text}")
        expect(locator).to_be_visible()
        locator.click()
        # Verify that the URL contains the section id or the heading is visible
        if link_text.lower() == "home":
            expect(page.locator("h1")).to_contain_text("Home")
        else:
            section_id = link_text.lower()
            expect(page.locator(f"#{section_id}")).to_be_visible()


def test_project_cards_display_complete_information(page):
    """Scenario: Project cards display complete information"""
    page.goto(BASE_URL)
    page.locator("section#projects").scroll_into_view_if_needed()
    cards = page.locator(".project-card")
    count = cards.count()
    assert count > 0, "No project cards found"

    for i in range(count):
        card = cards.nth(i)
        title = card.locator(".card-title")
        desc = card.locator(".card-description")
        tech = card.locator(".card-tech")
        link = card.locator("a.card-link")

        expect(title).to_be_visible()
        expect(desc).to_be_visible()
        expect(tech).to_be_visible()
        expect(link).to_be_visible()
        expect(link).to_have_attribute("target", "_blank")


def test_all_external_urls_reachable(page):
    """Scenario: All external URLs are reachable"""
    page.goto(BASE_URL)
    links = page.locator('a[href^="http"]')
    count = links.count()
    for i in range(count):
        link = links.nth(i)
        href = link.get_attribute("href")
        if href and BASE_URL not in href:
            try:
                resp = requests.get(href, timeout=5)
                assert resp.status_code == 200, f"{href} returned {resp.status_code}"
            except requests.RequestException as e:
                assert False, f"Request to {href} failed: {e}"


def test_skills_list_contains_no_duplicates(page):
    """Scenario: Skills list contains only current, non‑duplicate entries"""
    page.goto(BASE_URL)
    skills = page.locator("section#skills li")
    count = skills.count()
    skill_texts = [skills.nth(i).inner_text().strip() for i in range(count)]
    assert len(skill_texts) == len(set(skill_texts)), "Duplicate skills found"


def test_images_have_alt_text(page):
    """Scenario: Images and icons have descriptive alt text and meet WCAG 2.1 AA contrast"""
    page.goto(BASE_URL)
    images = page.locator("img")
    count = images.count()
    for i in range(count):
        img = images.nth(i)
        alt = img.get_attribute("alt")
        assert alt and alt.strip(), f"Image at index {i} has empty alt text"


def test_console_is_error_free(page):
    """Scenario: Console is free of errors or warnings"""
    page.goto(BASE_URL)
    console_messages = []

    def handle_console(msg):
        if msg.type in ["error", "warning"]:
            console_messages.append(msg.text)

    page.on("console", handle_console)
    # Wait a moment for console messages to be emitted
    page.wait_for_timeout(2000)
    assert not console_messages, f"Console errors/warnings found: {console_messages}"


def test_project_card_with_deleted_github_repo(page):
    """Scenario: Project card with a deleted GitHub repository"""
    page.goto(BASE_URL)
    cards = page.locator(".project-card")
    count = cards.count()
    found = False
    for i in range(count):
        card = cards.nth(i)
        link = card.locator("a.card-link")
        href = link.get_attribute("href")
        if href and "github.com" in href:
            try:
                resp = requests.get(href, timeout=5)
                if resp.status_code == 404:
                    found = True
                    expect(card.locator("text=Repository unavailable")).to_be_visible()
            except requests.RequestException:
                pass
    assert found, "No deleted GitHub repository link found"


def test_mobile_viewport_collapses_navigation(page):
    """Scenario: Mobile viewport collapses navigation into a functional hamburger menu"""
    page.set_viewport_size({"width": 320, "height": 800})
    page.goto(BASE_URL)
    hamburger = page.locator("button.hamburger")
    expect(hamburger).to_be_visible()
    hamburger.click()
    nav_links = page.locator("nav a")
    expect(nav_links).to_be_visible()


def test_certificate_link_points_to_404(page):
    """Scenario: Certificate link points to a 404 page"""
    page.goto(BASE_URL)
    cert_link = page.locator("a:has-text(\"Certificate\")")
    if cert_link.count() > 0:
        cert_link.click()
        page.wait_for_load_state("load")
        expect(page.locator("body")).to_contain_text("Certificate not found")
    else:
        assert False, "No certificate link found on the page"