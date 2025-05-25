# Alpaca Positions Testing & Portfolio Analysis

This document describes the comprehensive Alpaca positions testing and portfolio analysis functionality that has been implemented.

## Overview

The Alpaca positions testing system provides multiple layers of functionality:

1. **Basic positions retrieval and display**
2. **Comprehensive portfolio analysis with risk metrics**
3. **Individual position risk assessment**
4. **Configuration integration with the AppConfig system**

## Available Test Scripts

### 1. `exchanges/test_alpaca_client.py`
**Primary Alpaca client test with comprehensive positions analysis**

Features:
- Account information retrieval
- Detailed positions listing with P&L analysis
- Portfolio summary with totals
- Current market price fetching
- Test trade placement (small amounts)
- Order management testing

### 2. `test_alpaca_positions.py`
**Focused positions testing with advanced portfolio analysis**

Features:
- Account overview with key metrics
- Detailed position analysis with formatted output
- Portfolio summary with performance metrics
- Individual position risk analysis
- Best/worst performer identification
- Risk level assessment for each position

### 3. `simple_positions_test.py`
**Simple diagnostic test for basic connectivity and positions**

Features:
- Step-by-step testing with clear progress indicators
- Basic error handling and diagnostics
- Minimal output for quick verification
- Credential validation

### 4. `example_config_usage.py`
**Comprehensive configuration system demonstration**

Features:
- Configuration system testing
- Multi-component integration (Alpaca, LLM, Data clients)
- Portfolio analysis integration
- Environment variable validation

## New Alpaca Client Methods

### Portfolio Analysis Methods

#### `get_portfolio_summary() -> Dict`
Returns comprehensive portfolio analysis including:
- **Account metrics**: Cash, buying power, portfolio value
- **Position metrics**: Total market value, unrealized P&L, cost basis
- **Performance metrics**: Total return percentage, best/worst performers
- **Individual position data**: Detailed breakdown by symbol

Example return structure:
```python
{
    'account': {
        'cash': 10000.0,
        'buying_power': 15000.0,
        'portfolio_value': 25000.0,
        'status': 'ACTIVE'
    },
    'positions': {
        'count': 3,
        'total_market_value': 15000.0,
        'total_unrealized_pl': 1500.0,
        'total_cost_basis': 13500.0,
        'by_symbol': {
            'BTCUSD': {
                'qty': 0.5,
                'market_value': 10000.0,
                'unrealized_pl': 1000.0,
                'return_pct': 11.11,
                'cost_basis': 9000.0
            }
        }
    },
    'performance': {
        'total_return_pct': 11.11,
        'best_performer': {
            'symbol': 'BTCUSD',
            'return_pct': 15.5,
            'unrealized_pl': 1000.0
        }
    }
}
```

#### `analyze_position_risk(symbol: str) -> Dict`
Analyzes risk metrics for a specific position:
- **Price analysis**: Current vs entry price comparison
- **Risk classification**: Low/Medium/High based on price movement
- **Risk flags**: Profitability, high risk, attention needed indicators

Example return:
```python
{
    'symbol': 'BTCUSD',
    'current_price': 45000.0,
    'entry_price': 40000.0,
    'market_value': 22500.0,
    'unrealized_pl': 2500.0,
    'price_change_pct': 12.5,
    'risk_level': 'Medium',
    'analysis': {
        'is_profitable': True,
        'is_high_risk': False,
        'needs_attention': False
    }
}
```

#### `list_positions() -> list`
Enhanced positions method that returns a clean list format with proper error handling.

## Configuration Integration

All test scripts integrate with the `AppConfig` system:

```python
from config.config import AppConfig

config = AppConfig()
alpaca_config = config.get_alpaca_config()

client = AlpacaCryptoTrading(
    api_key=alpaca_config['api_key'],
    api_secret=alpaca_config['api_secret'],
    base_url=alpaca_config['base_url']
)
```

## Risk Assessment Features

### Risk Levels
- **Low Risk**: Price change < 10%
- **Medium Risk**: Price change 10-20%
- **High Risk**: Price change > 20%

### Risk Indicators
- **is_profitable**: Position has positive unrealized P&L
- **is_high_risk**: Price movement > 15%
- **needs_attention**: Price movement > 25%

## Usage Examples

### Basic Positions Check
```bash
python3 simple_positions_test.py
```

### Comprehensive Portfolio Analysis
```bash
python3 test_alpaca_positions.py
```

### Configuration Testing
```bash
python3 example_config_usage.py
```

### Full Trading Client Test
```bash
python3 exchanges/test_alpaca_client.py
```

## Error Handling

All test scripts include comprehensive error handling:
- **Import errors**: Missing dependencies or modules
- **Configuration errors**: Missing API keys or invalid credentials
- **API errors**: Network issues, authentication failures
- **Data parsing errors**: Invalid response formats

## Output Formatting

Consistent formatting across all scripts:
- **Currency values**: `$1,234.56` format
- **Percentages**: `+12.34%` format with sign
- **Status indicators**: ✅ ❌ ⚠️ 🚨 emojis for visual clarity
- **Structured output**: Clear section headers and hierarchical display

## Prerequisites

1. **Environment Variables**: Set in `.env` file
   ```
   ALPACA_API_KEY=your_key_here
   ALPACA_SECRET_KEY=your_secret_here
   ```

2. **Dependencies**: Ensure all required packages are installed
   ```bash
   pip install -r requirements.txt
   ```

3. **Account Setup**: Valid Alpaca account (paper or live trading)

## Troubleshooting

### Common Issues
1. **No positions found**: Account may be empty or using wrong environment (paper vs live)
2. **Authentication errors**: Check API keys in `.env` file
3. **Network errors**: Verify internet connection and Alpaca service status
4. **Import errors**: Ensure all dependencies are installed

### Debug Steps
1. Run `simple_positions_test.py` for basic connectivity check
2. Verify credentials with `example_config_usage.py`
3. Check API status at Alpaca's status page
4. Review error messages for specific issue details

This comprehensive testing system provides everything needed to verify Alpaca integration, analyze portfolio performance, and assess position risks within the crypto trading system.
