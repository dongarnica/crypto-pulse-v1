import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.alpaca_client import AlpacaCryptoTrading


@pytest.fixture
def alpaca_client():
    """Create an AlpacaCryptoTrading client for testing."""
    return AlpacaCryptoTrading(
        api_key="test_api_key",
        api_secret="test_api_secret",
        base_url="https://paper-api.alpaca.markets"
    )


@pytest.fixture
def mock_response():
    """Mock response data for API calls."""
    return {
        "account": {
            "id": "test_account",
            "status": "ACTIVE",
            "cash": "10000.00",
            "buying_power": "40000.00",
            "portfolio_value": "50000.00",
            "daytrade_count": 0
        },
        "order": {
            "id": "order123",
            "symbol": "BTCUSD",
            "qty": "1",
            "side": "buy",
            "status": "filled",
            "type": "market",
            "time_in_force": "gtc"
        },
        "positions": [
            {
                "symbol": "BTCUSD",
                "qty": "1.5",
                "market_value": "75000.00",
                "side": "long",
                "unrealized_pl": "5000.00"
            }
        ]
    }


class TestAlpacaCryptoTrading:
    """Test suite for AlpacaCryptoTrading class."""

    @patch('exchanges.alpaca_client.requests.Session.request')
    def test_get_account(self, mock_request, alpaca_client, mock_response):
        """Test get_account method."""
        mock_request.return_value.json.return_value = mock_response["account"]
        mock_request.return_value.raise_for_status.return_value = None
        
        account = alpaca_client.get_account()
        
        assert account["id"] == "test_account"
        assert account["status"] == "ACTIVE"
        assert account["cash"] == "10000.00"
        mock_request.assert_called_once()

    @patch('exchanges.alpaca_client.requests.Session.request')
    def test_list_positions(self, mock_request, alpaca_client, mock_response):
        """Test list_positions method."""
        mock_request.return_value.json.return_value = mock_response["positions"]
        mock_request.return_value.raise_for_status.return_value = None
        
        positions = alpaca_client.list_positions()
        
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTCUSD"
        assert positions[0]["qty"] == "1.5"
        mock_request.assert_called_once()

    @patch('exchanges.alpaca_client.requests.Session.request')
    def test_place_order(self, mock_request, alpaca_client, mock_response):
        """Test place_order method."""
        mock_request.return_value.json.return_value = mock_response["order"]
        mock_request.return_value.raise_for_status.return_value = None
        
        order = alpaca_client.place_order(
            symbol="BTCUSD",
            qty=1.0,
            side="buy",
            type="market"
        )
        
        assert order["id"] == "order123"
        assert order["symbol"] == "BTCUSD"
        assert order["qty"] == "1"
        assert order["side"] == "buy"
        assert order["status"] == "filled"
        mock_request.assert_called_once()

    @patch('exchanges.alpaca_client.requests.Session.request')
    def test_cancel_order(self, mock_request, alpaca_client):
        """Test cancel_order method."""
        mock_request.return_value.json.return_value = {}
        mock_request.return_value.raise_for_status.return_value = None
        
        result = alpaca_client.cancel_order(order_id="order123")
        
        assert result == {}
        mock_request.assert_called_once()

    @patch('exchanges.alpaca_client.requests.Session.request')
    def test_get_order(self, mock_request, alpaca_client, mock_response):
        """Test get_order method."""
        mock_request.return_value.json.return_value = mock_response["order"]
        mock_request.return_value.raise_for_status.return_value = None
        
        order = alpaca_client.get_order(order_id="order123")
        
        assert order["id"] == "order123"
        assert order["symbol"] == "BTCUSD"
        mock_request.assert_called_once()

    @patch('exchanges.alpaca_client.requests.Session.request')
    def test_get_last_trade(self, mock_request, alpaca_client):
        """Test get_last_trade method."""
        mock_trade_data = {
            "symbol": "BTCUSD",
            "price": 50000.0,
            "size": 0.5,
            "timestamp": "2023-01-01T12:00:00Z"
        }
        mock_request.return_value.json.return_value = mock_trade_data
        mock_request.return_value.raise_for_status.return_value = None
        
        trade = alpaca_client.get_last_trade("BTCUSD")
        
        assert trade["symbol"] == "BTCUSD"
        assert trade["price"] == 50000.0
        mock_request.assert_called_once()

    @patch('exchanges.alpaca_client.requests.Session.request')
    def test_get_last_quote(self, mock_request, alpaca_client):
        """Test get_last_quote method."""
        mock_quote_data = {
            "symbol": "BTCUSD",
            "bid": 49950.0,
            "ask": 50050.0,
            "timestamp": "2023-01-01T12:00:00Z"
        }
        mock_request.return_value.json.return_value = mock_quote_data
        mock_request.return_value.raise_for_status.return_value = None
        
        quote = alpaca_client.get_last_quote("BTCUSD")
        
        assert quote["symbol"] == "BTCUSD"
        assert quote["bid"] == 49950.0
        assert quote["ask"] == 50050.0
        mock_request.assert_called_once()

    def test_initialization(self):
        """Test client initialization."""
        client = AlpacaCryptoTrading(
            api_key="test_key",
            api_secret="test_secret",
            base_url="https://paper-api.alpaca.markets"
        )
        
        assert client.api_key == "test_key"
        assert client.api_secret == "test_secret"
        assert client.base_url == "https://paper-api.alpaca.markets"
        assert 'APCA-API-KEY-ID' in client.session.headers
        assert 'APCA-API-SECRET-KEY' in client.session.headers
