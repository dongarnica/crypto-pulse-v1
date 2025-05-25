import pytest
from exchanges.alpaca_client import AlpacaTradingClient

class MockTradingClient:
    def get_account(self):
        return {"id": "test_account", "status": "ACTIVE"}
    def submit_order(self, order_req):
        return {"id": "order123", "symbol": order_req.symbol, "qty": order_req.qty, "side": order_req.side, "status": "filled"}
    def cancel_order(self, order_id):
        return {"id": order_id, "status": "cancelled"}

@pytest.fixture
def alpaca_client(monkeypatch):
    # Patch TradingClient in AlpacaTradingClient to use the mock
    monkeypatch.setattr("exchanges.alpaca_client.TradingClient", lambda *a, **k: MockTradingClient())
    return AlpacaTradingClient(api_key="test", secret_key="test", paper=True)

def test_get_account(alpaca_client):
    account = alpaca_client.get_account()
    assert account["id"] == "test_account"
    assert account["status"] == "ACTIVE"

def test_submit_market_order(alpaca_client):
    order = alpaca_client.submit_market_order(symbol="BTCUSD", qty=1, side="buy")
    assert order["id"] == "order123"
    assert order["symbol"] == "BTCUSD"
    assert order["qty"] == 1
    assert order["side"].name == "BUY"
    assert order["status"] == "filled"

def test_cancel_order(alpaca_client):
    result = alpaca_client.cancel_order(order_id="order123")
    assert result["id"] == "order123"
    assert result["status"] == "cancelled"
