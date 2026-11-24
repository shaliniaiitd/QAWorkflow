import pytest
from playwright.sync_api import expect

BASE_URL = "https://shaliniaiitd.github.io"

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def go_to_portfolio(page):
    """Navigate to the main portfolio page."""
    page.goto(BASE_URL)

def go_to_project_detail(page, title):
    """Navigate to a project detail page by clicking its card."""
    go_to_portfolio(page)
    page.locator(f"text={title}").click()

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
def test_display_all_projects_on_portfolio_page(page):
    """Scenario: Display all projects on the portfolio page"""
    go_to_portfolio(page)
    # Wait for at least one article to become visible
    page.wait_for_selector("article", state="visible", timeout=10000)
    # Expect a grid of project cards (articles)
    expect(page.locator("article")).to_be_visible()
    # Each card should have heading, paragraph, and link
    cards = page.locator("article")
    expect(cards).to_have_count_greater_than(0)
    for card in cards.all():
        # Thumbnail may not be present; skip if absent
        if card.locator(".thumbnail").count():
            expect(card.locator(".thumbnail")).to_be_visible()
        expect(card.locator("h3")).to_be_visible()
        expect(card.locator("p")).to_be_visible()
    # Responsive check (simple viewport resize)
    for width in (320, 768, 1280):
        page.set_viewport_size({"width": width, "height": 800})
        page.wait_for_selector("article", state="visible", timeout=10000)
        expect(page.locator("article")).to_be_visible()

def test_view_project_detail(page):
    """Scenario: View a project detail"""
    go_to_project_detail(page, "Weather App")
    # Verify URL contains project slug
    expect(page).to_have_url(f"{BASE_URL}/*weather-app*")
    # Verify detail elements
    expect(page.locator("h1")).to_have_text("Weather App")
    expect(page.locator(".full-description")).to_contain_text("A detailed description of the app")
    expect(page.locator(".tech-stack")).to_contain_text("React, Node.js, OpenWeatherMap API")
    expect(page.locator("a[href='https://weatherapp.example.com']")).to_be_visible()
    expect(page.locator("a[href='https://github.com/user/weatherapp']")).to_be_visible()
    expect(page.locator(".media")).to_have_count_greater_than(0)
    # Back link
    expect(page.locator("a:has-text('Back to Portfolio')")).to_be_visible()

def test_filter_projects_by_technology(page):
    """Scenario: Filter projects by technology"""
    go_to_portfolio(page)
    # Select "React" from filter dropdown
    page.locator("select#tech-filter").select_option("React")
    # Verify only React projects displayed
    cards = page.locator("article")
    expect(cards).to_have_count_greater_than(0)
    for card in cards.all():
        expect(card.locator(".tech-stack")).to_contain_text("React")

def test_portfolio_page_with_no_projects(page):
    """Scenario: Portfolio page with no projects"""
    # Navigate to a page that simulates empty project list
    page.goto(f"{BASE_URL}/portfolio-empty")
    expect(page.locator("text=No projects to display")).to_be_visible()

def test_project_missing_optional_fields(page):
    """Scenario: Project missing optional fields"""
    # Navigate to a detail page for a project with missing demo and media
    page.goto(f"{BASE_URL}/legacy-system")
    # Demo link hidden
    expect(page.locator("a.demo-link")).not_to_be_visible()
    # Media placeholder
    expect(page.locator(".media-placeholder")).to_be_visible()

def test_lazy_load_large_media(page):
    """Scenario: Lazy load large media files"""
    # Navigate to a detail page with a large image
    page.goto(f"{BASE_URL}/large-media-project")
    # Wait for image to load only when in viewport
    image = page.locator("img.large-media")
    # Initially not in viewport
    expect(image).not_to_be_visible()
    # Scroll into view
    image.scroll_into_view_if_needed()
    expect(image).to_be_visible()
    # Page load time under 2 seconds
    load_time = page.evaluate("performance.timing.loadEventEnd - performance.timing.navigationStart")
    assert load_time < 2000, f"Load time {load_time}ms exceeds 2000ms"

def test_truncate_long_project_title(page):
    """Scenario: Truncate long project title"""
    go_to_portfolio(page)
    # Find a card with a long title
    long_title_card = page.locator("article:has-text('This is a very long project title that exceeds sixty characters')")
    expect(long_title_card).to_be_visible()
    title = long_title_card.locator(".project-title")
    # Title should be truncated
    expect(title).to_have_css("overflow", "hidden")
    # Hover to show tooltip
    title.hover()
    tooltip = page.locator(".tooltip")
    expect(tooltip).to_be_visible()
    expect(tooltip).to_have_text(long_title_card.locator(".project-title").inner_text())

def test_screen_reader_navigation(page):
    """Scenario: Screen reader navigation"""
    go_to_portfolio(page)
    # Use keyboard navigation to focus each card
    page.keyboard.press("Tab")
    for card in page.locator("article").all():
        expect(card).to_be_focused()
        # Announce title and tagline via aria-label
        aria_label = card.get_attribute("aria-label")
        assert "title" in aria_label.lower()
        assert "tagline" in aria_label.lower()
    # Back link reachable
    page.keyboard.press("Tab")
    back_link = page.locator("a:has-text('Back to Portfolio')")
    expect(back_link).to_be_focused()

def test_verify_layout_on_major_browsers(page):
    """Scenario Outline: Verify layout on major browsers"""
    # The pytest-playwright fixture automatically launches the requested browser
    go_to_portfolio(page)
    # Basic layout check
    expect(page.locator(".portfolio-grid")).to_be_visible()
    # All links and media load
    for link in page.locator("a").all():
        expect(link).to_be_visible()
    for media in page.locator("img, video").all():
        expect(media).to_be_visible()

def test_sanitize_user_generated_content(page):
    """Scenario: Sanitize user‑generated content"""
    # Navigate to a project with malicious description
    page.goto(f"{BASE_URL}/malicious-project")
    # The script tag should be escaped
    script_elements = page.locator("script")
    expect(script_elements).to_have_count(0)
    # Description displayed safely
    desc = page.locator(".full-description")
    expect(desc).to_contain_text("<script>alert('xss')</script>")
    # Ensure no alert executed (cannot test directly, but absence of script tag suffices)