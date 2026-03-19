import django_tables2 as tables
from django_tables2.utils import A
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from .models import PrecomputedMetrics

class ChangeColumn(tables.Column):
    """Custom column for percentage changes with color coding"""

    def render(self, value, record, bound_column):
        if value is None:
            return mark_safe('<span class="text-muted">-</span>')

        # Ensure value is a float
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            return mark_safe('<span class="text-muted">-</span>')

        color = 'text-success' if float_value >= 0 else 'text-danger'
        sign = '+' if float_value > 0 else ''

        # Format the number properly
        formatted_value = f"{sign}{float_value:.2f}%"

        return format_html(
            '<span class="{}">{}</span>',
            color,
            formatted_value
        )

class MarketCapColumn(tables.Column):
    """Format market cap in billions/millions"""

    def render(self, value, record, bound_column):
        if value is None:
            return mark_safe('<span class="text-muted">-</span>')

        try:
            float_value = float(value)
        except (TypeError, ValueError):
            return mark_safe('<span class="text-muted">-</span>')

        if float_value >= 1e9:
            formatted = f'{float_value/1e9:.2f}B'
        elif float_value >= 1e6:
            formatted = f'{float_value/1e6:.2f}M'
        else:
            formatted = f'{float_value:.2f}'

        return mark_safe(f'<span>{formatted}</span>')

class SymbolMetricsTable(tables.Table):
    ticker = tables.Column(
        accessor='symbol__ticker',
        verbose_name='Symbol',
        linkify=lambda record: f"/markets/chart/?ticker={record.symbol.ticker}",
        attrs={'td': {'class': 'fw-bold'}}
    )
    name = tables.Column(
        accessor='symbol__name',
        verbose_name='Company Name',
        attrs={'td': {'class': 'text-truncate', 'style': 'max-width: 200px;'}}
    )
    market_cap = MarketCapColumn(
        accessor='symbol__market_cap',
        verbose_name='Market Cap',
        attrs={'td': {'class': 'text-end'}}
    )
    sector = tables.Column(
        accessor='symbol__sector__name',
        verbose_name='Sector',
        default='-'
    )
    industry = tables.Column(
        accessor='symbol__industry__name',
        verbose_name='Industry',
        default='-',
        attrs={'td': {'class': 'text-truncate', 'style': 'max-width: 150px;'}}
    )
    current_price = tables.Column(
        verbose_name='Price',
        attrs={'td': {'class': 'text-end fw-bold'}}
    )

    change_1d = ChangeColumn(verbose_name='%Change 1D', attrs={'td': {'class': 'text-end'}})
    change_1w = ChangeColumn(verbose_name='%Change 1W', attrs={'td': {'class': 'text-end'}})
    change_2w = ChangeColumn(verbose_name='%Change 2W', attrs={'td': {'class': 'text-end'}})
    change_1m = ChangeColumn(verbose_name='%Change 1M', attrs={'td': {'class': 'text-end'}})
    change_3m = ChangeColumn(verbose_name='%Change 3M', attrs={'td': {'class': 'text-end'}})
    change_6m = ChangeColumn(verbose_name='%Change 6M', attrs={'td': {'class': 'text-end'}})
    change_1y = ChangeColumn(verbose_name='%Change 1Y', attrs={'td': {'class': 'text-end'}})

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
            'tbody': {'class': 'align-middle'}
        }
        order_by = 'ticker'
        per_page = 25

