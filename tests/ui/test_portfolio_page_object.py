from playwright.sync_api import Page, expect

from tests.ui.pages.portfolio_page import PortfolioPage


PORTFOLIO_FIXTURE = """<!doctype html>
<html lang="en"><head><title>Portfolio fixture</title></head><body>
<header><nav aria-label="Main"><a href="#projects">Projects</a></nav></header>
<main><h1>QA Engineer</h1><section id="projects" aria-labelledby="projects-title">
<h2 id="projects-title">Projects</h2><article aria-label="QA Workflow">
<h3>QA Workflow</h3><p>Regression and browser automation project.</p></article>
</section></main></body></html>"""


def test_projects_navigation_and_card(page: Page):
    page.set_content(PORTFOLIO_FIXTURE)
    portfolio = PortfolioPage(page)
    portfolio.open_projects("about:blank")
    expect(portfolio.heading("Projects")).to_be_visible()
    expect(portfolio.project_card("QA Workflow")).to_contain_text("Regression")
