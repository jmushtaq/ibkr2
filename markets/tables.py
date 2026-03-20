import django_tables2 as tables
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from .models import PrecomputedMetrics

class SymbolMetricsTable(tables.Table):
    # Define columns with explicit accessors
    ticker = tables.Column(
        accessor='symbol.ticker',
        verbose_name='Symbol',
        orderable=True,
    )

    name = tables.Column(
        accessor='symbol.name',
        verbose_name='Company Name',
        orderable=True,
    )

    market_cap = tables.Column(
        accessor='symbol.market_cap',
        verbose_name='Market Cap (B)',
        orderable=True,
    )

    sector = tables.Column(
        accessor='symbol.sector.name',
        verbose_name='Sector',
        orderable=True,
    )

    industry = tables.Column(
        accessor='symbol.industry.name',
        verbose_name='Industry',
        orderable=True,
    )

    current_price = tables.Column(
        accessor='current_price',
        verbose_name='Price',
        orderable=True,
    )

    change_1d = tables.Column(
        accessor='change_1d',
        verbose_name='1D %',
        orderable=True,
    )

    change_1w = tables.Column(
        accessor='change_1w',
        verbose_name='1W %',
        orderable=True,
    )

    change_2w = tables.Column(
        accessor='change_2w',
        verbose_name='2W %',
        orderable=True,
    )

    change_1m = tables.Column(
        accessor='change_1m',
        verbose_name='1M %',
        orderable=True,
    )

    change_3m = tables.Column(
        accessor='change_3m',
        verbose_name='3M %',
        orderable=True,
    )

    change_6m = tables.Column(
        accessor='change_6m',
        verbose_name='6M %',
        orderable=True,
    )

    change_1y = tables.Column(
        accessor='change_1y',
        verbose_name='1Y %',
        orderable=True,
    )

    def render_ticker(self, value, record):
        """Render ticker as a link to the chart page"""
        if value is None:
            return '-'
        return format_html('<a href="/markets/chart/?ticker={}" target="_blank">{}</a>', value, value)

    def render_name(self, value, record):
        """Render company name"""
        if value is None or value == '':
            return record.symbol.ticker if record and hasattr(record, 'symbol') else '-'
        return value

    def render_market_cap(self, value, record):
        if value is None:
            return '-'
        return str(value)

    def render_sector(self, value, record):
        if value is None:
            return '-'
        return str(value)

    def render_industry(self, value, record):
        if value is None:
            return '-'
        return str(value)

    def render_current_price(self, value, record):
        if value is None:
            return '-'
        return str(value)

    def render_change_1d(self, value, record):
        if value is None:
            return '-'
        return str(value)

    def render_change_1w(self, value, record):
        if value is None:
            return '-'
        return str(value)

    def render_change_2w(self, value, record):
        if value is None:
            return '-'
        return str(value)

    def render_change_1m(self, value, record):
        if value is None:
            return '-'
        return str(value)

    def render_change_3m(self, value, record):
        if value is None:
            return '-'
        return str(value)

    def render_change_6m(self, value, record):
        if value is None:
            return '-'
        return str(value)

    def render_change_1y(self, value, record):
        if value is None:
            return '-'
        return str(value)

    class Meta:
        model = PrecomputedMetrics
        template_name = "django_tables2/bootstrap5.html"
        fields = [
            'ticker', 'name', 'market_cap', 'sector', 'industry',
            'current_price', 'change_1d', 'change_1w', 'change_2w',
            'change_1m', 'change_3m', 'change_6m', 'change_1y',
        ]
        attrs = {
            'class': 'table table-striped table-hover table-sm',
            'thead': {'class': 'table-dark'},
            'id': 'market-data-table',
        }
        order_by = 'ticker'
        per_page = 25

