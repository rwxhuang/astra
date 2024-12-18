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


@patch("selenium.webdriver.Chrome")
def test_run_method(mock_webdriver, scraper):
    """Test the run method of the scraper."""
    mock_driver = scraper.driver

    # Mock the get and find_element methods
    mock_driver.get = MagicMock()
    mock_search_button = MagicMock()
    mock_driver.find_element.return_value = mock_search_button

    scraper.run()

    mock_driver.get.assert_called_once_with(
        "https://techport.nasa.gov/advancedSearch")
    mock_driver.find_element.assert_called_once_with(
        By.CLASS_NAME, '_large_s5jgh_128')
    mock_search_button.click.assert_called_once()
