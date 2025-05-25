import pytest
from data.crypto_data_client import CryptoMarketDataClient

def test_get_crypto_data_success(monkeypatch):
    # Mock response data
    mock_response = {
        "bitcoin": {
            "usd": 50000,
            "usd_market_cap": 900000000000,
            "usd_24h_vol": 35000000000,
            "usd_24h_change": 2.5,
            "last_updated_at": 1620000000
        },
        "ethereum": {
            "usd": 4000,
            "usd_market_cap": 400000000000,
            "usd_24h_vol": 20000000000,
            "usd_24h_change": 1.2,
            "last_updated_at": 1620000000
        }
    }

    class MockResp:
        def raise_for_status(self):
            pass
        def json(self):
            return mock_response

    def mock_get(*args, **kwargs):
        return MockResp()

    import requests
    monkeypatch.setattr(requests, "get", mock_get)
    data = CryptoMarketDataClient.get_crypto_data()
    assert data == mock_response
    assert "bitcoin" in data
    assert data["bitcoin"]["usd"] == 50000
    assert "ethereum" in data
    assert data["ethereum"]["usd"] == 4000

def test_get_crypto_data_failure(monkeypatch):
    def mock_get(*args, **kwargs):
        raise Exception("API error")
    import requests
    monkeypatch.setattr(requests, "get", mock_get)
    data = CryptoMarketDataClient.get_crypto_data()
    assert data is None
