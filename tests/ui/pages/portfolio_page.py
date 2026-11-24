from playwright.sync_api import Page, expect

from .base_page import BasePage


class PortfolioPage(BasePage):
    """Page object for the shared portfolio navigation and project cards."""

    def __init__(self, page: Page):
        super().__init__(page)
        self.projects_link = page.get_by_role("link", name="Projects", exact=True)

    def open_projects(self, url: str) -> None:
        self.open(url)
        expect(self.projects_link).to_be_visible()
        self.projects_link.click()

    def project_card(self, name: str):
        return self.page.get_by_role("article", name=name, exact=False)
