import pytest
from playwright.sync_api import expect

# Updated base URL to point directly to the portfolio page
BASE_URL = "https://shaliniaiitd.github.io/portfolio"

@pytest.fixture(autouse=True)
def load_portfolio_page(page):
    """Background: load the portfolio page on a typical broadband connection."""
    # Ensure navigation completes before proceeding
    page.goto(BASE_URL, wait_until="load")
    # Assume the portfolio content is within a main element with id 'portfolio'
    expect(page.locator("#portfolio")).to_be_visible()

def test_view_list_of_projects(page):
    """Scenario: View the list of projects"""
    grid = page.locator(".project-grid")
    expect(grid).to_be_visible()
    thumbnails = grid.locator(".project-thumbnail")
    expect(thumbnails).to_have_count(lambda count: count > 0)
    titles = thumbnails.locator(".project-title")
    taglines = thumbnails.locator(".project-tagline")
    expect(titles).to_have_count(lambda count: count > 0)
    expect(taglines).to_have_count(lambda count: count > 0)
    # Responsive check: viewport width 320 (mobile)
    page.set_viewport_size({"width": 320, "height": 800})
    expect(grid).to_be_visible()
    # Tablet
    page.set_viewport_size({"width": 768, "height": 1024})
    expect(grid).to_be_visible()
    # Desktop
    page.set_viewport_size({"width": 1280, "height": 800})
    expect(grid).to_be_visible()

def test_open_project_detail_view(page):
    """Scenario: Open a project detail view"""
    thumbnail = page.locator(".project-thumbnail").first
    thumbnail.click()
    modal = page.locator(".project-modal")
    expect(modal).to_be_visible()
    description = modal.locator(".project-description")
    expect(description).to_be_visible()
    tech_list = modal.locator(".project-tech-list")
    expect(tech_list).to_be_visible()
    github_link = modal.locator("a[href*='github.com']")
    live_demo_link = modal.locator("a[href*='demo']")
    expect(github_link).to_be_visible()
    expect(live_demo_link).to_be_visible()

def test_sort_projects_by_category(page):
    """Scenario: Sort projects by category"""
    filter_button = page.locator("button#category-filter")
    filter_button.click()
    option = page.locator("li[data-category='Web Development']")
    option.click()
    thumbnails = page.locator(".project-thumbnail")
    expect(thumbnails).to_have_count(lambda count: count > 0)
    titles = thumbnails.locator(".project-title")
    title_texts = titles.all_text_contents()
    assert title_texts == sorted(title_texts), "Projects are not sorted alphabetically"

def test_filter_projects_by_tech_stack(page):
    """Scenario: Filter projects by tech stack"""
    filter_button = page.locator("button#tech-stack-filter")
    filter_button.click()
    option = page.locator("li[data-tech='React']")
    option.click()
    thumbnails = page.locator(".project-thumbnail")
    expect(thumbnails).to_have_count(lambda count: count > 0)
    for thumb in thumbnails.all():
        techs = thumb.locator(".project-tech-list").all_text_contents()
        assert "React" in techs, "React not found in tech stack"

def test_navigate_projects_with_keyboard(page):
    """Scenario: Navigate projects with keyboard"""
    first_thumb = page.locator(".project-thumbnail").first
    first_thumb.focus()
    for _ in range(5):
        page.keyboard.press("Tab")
    focused = page.evaluate("document.activeElement")
    assert focused is not None, "No element focused after Tab"
    page.keyboard.press("Enter")
    modal = page.locator(".project-modal")
    expect(modal).to_be_visible()

def test_accessibility_of_project_items(page):
    """Scenario: Accessibility of project items"""
    thumbnails = page.locator(".project-thumbnail")
    for thumb in thumbnails.all():
        name = thumb.get_attribute("aria-label")
        assert name is not None and len(name) > 0, "Accessible name missing"
        img = thumb.locator("img")
        alt = img.get_attribute("alt")
        assert alt is not None and len(alt) > 0, "Alt text missing"

def test_placeholder_image_for_missing_thumbnails(page):
    """Scenario: Placeholder image for missing thumbnails"""
    # Assume a project with id 'missing-thumb' has no image
    thumb = page.locator("#missing-thumb .project-thumbnail")
    img = thumb.locator("img")
    expect(img).to_have_attribute("src", "/images/placeholder.png")

def test_text_truncation_for_long_titles(page):
    """Scenario: Text truncation for long titles"""
    thumb = page.locator(".project-thumbnail").filter(has_text=lambda text: len(text) > 50).first
    title = thumb.locator(".project-title")
    expect(title).to_have_css("overflow", "hidden")
    expect(title).to_have_css("text-overflow", "ellipsis")

def test_disabled_link_for_invalid_urls(page):
    """Scenario: Disabled link for invalid URLs"""
    thumb = page.locator(".project-thumbnail").filter(has_text="Invalid URL Project").first
    thumb.click()
    modal = page.locator(".project-modal")
    expect(modal).to_be_visible()
    github_link = modal.locator("a[href*='github.com']")
    expect(github_link).to_be_disabled()
    tooltip = github_link.locator(".tooltip")
    expect(tooltip).to_have_text("Invalid GitHub URL")

def test_graceful_fallback_for_no_javascript_users(page):
    """Scenario: Graceful fallback for no-JavaScript users"""
    # Disable JavaScript
    page.context.set_java_script_enabled(False)
    page.goto(BASE_URL, wait_until="load")
    grid = page.locator(".project-grid")
    expect(grid).to_be_visible()
    links = grid.locator("a")
    expect(links).to_have_count(lambda count: count > 0)
    page.context.set_java_script_enabled(True)

def test_performance_within_2_seconds(page):
    """Scenario: Performance within 2 seconds"""
    with page.expect_navigation() as nav:
        page.goto(BASE_URL, wait_until="load")
    nav.wait_for()
    load_time = page.evaluate("performance.timing.loadEventEnd - performance.timing.navigationStart")
    assert load_time < 2000, f"Load time {load_time}ms exceeds 2000ms"

def test_no_broken_images_or_links(page):
    """Scenario: No broken images or links"""
    images = page.locator("img")
    for img in images.all():
        src = img.get_attribute("src")
        if src:
            response = page.request.get(src)
            assert response.status == 200, f"Image {src} failed to load"
    links = page.locator("a[href]")
    for link in links.all():
        href = link.get_attribute("href")
        if href and href.startswith("http"):
            response = page.request.get(href)
            assert response.status == 200, f"Link {href} returned {response.status}"

def test_unique_navigation_identifiers_for_duplicate_titles(page):
    """Scenario: Unique navigation identifiers for duplicate titles"""
    thumb1 = page.locator(".project-thumbnail").filter(has_text="Duplicate Title").nth(0)
    thumb2 = page.locator(".project-thumbnail").filter(has_text="Duplicate Title").nth(1)
    thumb1.click()
    expect(page).to_have_url(lambda url: "#duplicate-title-1" in url)
    page.go_back()
    thumb2.click()
    expect(page).to_have_url(lambda url: "#duplicate-title-2" in url)

def test_negative_path_project_with_missing_description(page):
    """Scenario: Negative path – project with missing description"""
    thumb = page.locator(".project-thumbnail").filter(has_text="No Description Project").first
    thumb.click()
    modal = page.locator(".project-modal")
    expect(modal).to_be_visible()
    description = modal.locator(".project-description")
    expect(description).to_have_text("No description available")

def test_negative_path_external_link_broken(page):
    """Scenario: Negative path – external link becomes broken"""
    thumb = page.locator(".project-thumbnail").filter(has_text="Broken Demo Project").first
    thumb.click()
    modal = page.locator(".project-modal")
    expect(modal).to_be_visible()
    live_demo = modal.locator("a[href*='broken-demo']")
    live_demo.click()
    error_msg = page.locator(".error-message")
    expect(error_msg).to_have_text("Link is unavailable")
    page.go_back()
    page.go_back()