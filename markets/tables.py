import django_tables2 as tables
from django.utils.html import format_html
from .models import PrecomputedMetrics

class SymbolMetricsTable(tables.Table):
    ticker = tables.Column(
        accessor='symbol__ticker',
        verbose_name='Symbol',
    )
    # ... other columns ...

    def render_ticker(self, value):
        """Render ticker as a link to the chart page"""
        return format_html(
            '<a href="/markets/chart/?ticker={}" target="_blank">{}</a>',
            value, value
        )

    class Meta:
        model = PrecomputedMetrics
        template_name = "django_tables2/bootstrap5.html"
        fields = [
            'ticker', 'name', 'market_cap', 'sector', 'industry',
            'current_price', 'change_1d', 'change_1w', 'change_2w',
            'change_1m', 'change_3m', 'change_6m', 'change_1y'
        ]
        attrs = {
            'class': 'table table-striped table-hover table-sm',
            'thead': {'class': 'table-dark'},
        }
        order_by = 'ticker'
        per_page = 25
