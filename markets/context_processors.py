from django.conf import settings

def dashboard_settings(request):
    """Add dashboard settings to template context"""
    return {
        'default_columns': getattr(settings, 'MARKET_DASHBOARD_DEFAULT_COLUMNS', [
            'ticker', 'name', 'market_cap', 'sector', 'industry',
            'current_price', 'change_1d', 'change_1w', 'change_1m', 'change_1y'
        ]),
        'all_columns': [
            # Basic
            {'id': 'ticker', 'name': 'Symbol', 'category': 'Basic'},
            {'id': 'name', 'name': 'Company Name', 'category': 'Basic'},
            {'id': 'market_cap', 'name': 'Market Cap (B)', 'category': 'Basic'},
            {'id': 'sector', 'name': 'Sector', 'category': 'Basic'},
            {'id': 'industry', 'name': 'Industry', 'category': 'Basic'},

            # Price
            {'id': 'current_price', 'name': 'Price', 'category': 'Price'},

            # Historical Returns
            {'id': 'change_1d', 'name': '1D %', 'category': 'Historical Returns'},
            {'id': 'change_1w', 'name': '1W %', 'category': 'Historical Returns'},
            {'id': 'change_2w', 'name': '2W %', 'category': 'Historical Returns'},
            {'id': 'change_1m', 'name': '1M %', 'category': 'Historical Returns'},
            {'id': 'change_3m', 'name': '3M %', 'category': 'Historical Returns'},
            {'id': 'change_6m', 'name': '6M %', 'category': 'Historical Returns'},
            {'id': 'change_1y', 'name': '1Y %', 'category': 'Historical Returns'},

            # Forward Max Rise
            {'id': 'fwd_max_rise_1d', 'name': 'Forward Max Rise 1D', 'category': 'Forward Max Rise'},
            {'id': 'fwd_max_rise_1w', 'name': 'Forward Max Rise 1W', 'category': 'Forward Max Rise'},
            {'id': 'fwd_max_rise_2w', 'name': 'Forward Max Rise 2W', 'category': 'Forward Max Rise'},
            {'id': 'fwd_max_rise_1m', 'name': 'Forward Max Rise 1M', 'category': 'Forward Max Rise'},
            {'id': 'fwd_max_rise_3m', 'name': 'Forward Max Rise 3M', 'category': 'Forward Max Rise'},
            {'id': 'fwd_max_rise_6m', 'name': 'Forward Max Rise 6M', 'category': 'Forward Max Rise'},
            {'id': 'fwd_max_rise_1y', 'name': 'Forward Max Rise 1Y', 'category': 'Forward Max Rise'},

            # Forward Max Drop
            {'id': 'fwd_max_drop_1d', 'name': 'Forward Max Drop 1D', 'category': 'Forward Max Drop'},
            {'id': 'fwd_max_drop_1w', 'name': 'Forward Max Drop 1W', 'category': 'Forward Max Drop'},
            {'id': 'fwd_max_drop_2w', 'name': 'Forward Max Drop 2W', 'category': 'Forward Max Drop'},
            {'id': 'fwd_max_drop_1m', 'name': 'Forward Max Drop 1M', 'category': 'Forward Max Drop'},
            {'id': 'fwd_max_drop_3m', 'name': 'Forward Max Drop 3M', 'category': 'Forward Max Drop'},
            {'id': 'fwd_max_drop_6m', 'name': 'Forward Max Drop 6M', 'category': 'Forward Max Drop'},
            {'id': 'fwd_max_drop_1y', 'name': 'Forward Max Drop 1Y', 'category': 'Forward Max Drop'},

            # Risk Metrics
            {'id': 'fwd_volatility_1m', 'name': 'Volatility 1M', 'category': 'Risk Metrics'},
            {'id': 'fwd_volatility_3m', 'name': 'Volatility 3M', 'category': 'Risk Metrics'},
            {'id': 'fwd_volatility_6m', 'name': 'Volatility 6M', 'category': 'Risk Metrics'},
            {'id': 'fwd_sharpe_ratio', 'name': 'Sharpe Ratio', 'category': 'Risk Metrics'},
            {'id': 'fwd_max_drawdown', 'name': 'Max Drawdown', 'category': 'Risk Metrics'},
            {'id': 'fwd_drawdown_duration', 'name': 'Drawdown Duration (Days)', 'category': 'Risk Metrics'},

            # Future columns (placeholder)
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

