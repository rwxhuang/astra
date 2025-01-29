import pytest
from unittest.mock import patch, MagicMock
from selenium.webdriver.common.by import By
from src import TechportScraper


@pytest.fixture
def scraper():
    """Fixture to initialize the TechportScraper."""
    with patch("selenium.webdriver.Chrome") as MockWebDriver:
        mock_driver = MockWebDriver.return_value
        scraper = TechportScraper(search_input="test input")
        scraper.driver = mock_driver
        yield scraper
        scraper.driver.quit()


@patch("selenium.webdriver.Chrome")
def test_scraper_initialization(mock_webdriver):
    """Test the initialization of the scraper."""
    mock_driver = mock_webdriver.return_value
    scraper = TechportScraper(search_input="test input")

    assert scraper.search_input == "test input"
    assert scraper.driver == mock_driver
