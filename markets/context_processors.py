# markets/context_processors.py
from django.conf import settings

def dashboard_settings(request):
    """Add dashboard settings to template context"""
    return {
        'default_columns': getattr(settings, 'MARKET_DASHBOARD_DEFAULT_COLUMNS', [
            'ticker', 'name', 'market_cap', 'sector', 'industry',
            'current_price', 'change_1d', 'change_1w', 'change_1m', 'change_1y'
        ]),
        'all_columns': [
            {'id': 'ticker', 'name': 'Symbol', 'category': 'Basic'},
            {'id': 'name', 'name': 'Company Name', 'category': 'Basic'},
            {'id': 'market_cap', 'name': 'Market Cap (B)', 'category': 'Basic'},
            {'id': 'sector', 'name': 'Sector', 'category': 'Basic'},
            {'id': 'industry', 'name': 'Industry', 'category': 'Basic'},
            {'id': 'current_price', 'name': 'Price', 'category': 'Price'},
            {'id': 'change_1d', 'name': '1D %', 'category': 'Returns'},
            {'id': 'change_1w', 'name': '1W %', 'category': 'Returns'},
            {'id': 'change_2w', 'name': '2W %', 'category': 'Returns'},
            {'id': 'change_1m', 'name': '1M %', 'category': 'Returns'},
            {'id': 'change_3m', 'name': '3M %', 'category': 'Returns'},
            {'id': 'change_6m', 'name': '6M %', 'category': 'Returns'},
            {'id': 'change_1y', 'name': '1Y %', 'category': 'Returns'},
            {'id': 'volume', 'name': 'Volume', 'category': 'Volume'},
            {'id': 'avg_volume_20d', 'name': 'Avg Vol (20D)', 'category': 'Volume'},
            {'id': 'pe_ratio', 'name': 'P/E', 'category': 'Fundamentals'},
            {'id': 'dividend_yield', 'name': 'Div Yield %', 'category': 'Fundamentals'},
            {'id': 'week_52_high', 'name': '52W High', 'category': 'Technical'},
            {'id': 'week_52_low', 'name': '52W Low', 'category': 'Technical'},
            {'id': 'relative_volume', 'name': 'Rel Volume', 'category': 'Technical'},
            {'id': 'atr', 'name': 'ATR', 'category': 'Technical'},
            {'id': 'rsi', 'name': 'RSI', 'category': 'Technical'},
        ],
    }

