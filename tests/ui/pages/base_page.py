from playwright.sync_api import Page


class BasePage:
    """Shared browser actions for page objects; prefer accessible locators."""

    def __init__(self, page: Page):
        self.page = page

    def open(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded")

    def heading(self, name: str):
        return self.page.get_by_role("heading", name=name, exact=True)
