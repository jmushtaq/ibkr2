# markets/templatetags/market_tags.py
from django import template
from django.utils.safestring import mark_safe
import json

register = template.Library()

@register.simple_tag
def get_column_visibility_settings():
    """Generate JavaScript settings for column visibility"""

    # Define all available columns with their categories
    all_columns = [
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

    ]

    # Default columns from settings
    default_columns = [
                'ticker', 'name', 'market_cap', 'sector', 'industry',
                'current_price', 'change_1d', 'change_1w', 'change_1m', 'change_1y'

    ]

    settings_json = json.dumps({
                'all_columns': all_columns,
                'default_columns': default_columns,

    })

    return mark_safe(f'<script>window.columnSettings = {settings_json};</script>')
