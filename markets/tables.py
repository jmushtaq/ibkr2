import django_tables2 as tables
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from .models import PrecomputedMetrics

class SymbolMetricsTable(tables.Table):
    # Existing columns
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

    # Backward-looking returns
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

    # Forward-looking metrics - Max Rise
    fwd_change_1d = tables.Column(
        accessor='fwd_change_1d',
        verbose_name='Fwd 1D %',
        orderable=True,
        #visible=False,
    )

    fwd_change_1w = tables.Column(
        accessor='fwd_change_1w',
        verbose_name='Fwd 1W %',
        orderable=True,
        #visible=False,
    )

    fwd_change_2w = tables.Column(
        accessor='fwd_change_2w',
        verbose_name='Fwd 2W %',
        orderable=True,
        #visible=False,
    )

    fwd_change_1m = tables.Column(
        accessor='fwd_change_1m',
        verbose_name='Fwd 1M %',
        orderable=True,
        #visible=False,
    )

    fwd_change_3m = tables.Column(
        accessor='fwd_change_3m',
        verbose_name='Fwd 3M %',
        orderable=True,
        #visible=False,
    )

    fwd_change_6m = tables.Column(
        accessor='fwd_change_6m',
        verbose_name='Fwd 6M %',
        orderable=True,
        #visible=False,
    )

    fwd_change_1y = tables.Column(
        accessor='fwd_change_1y',
        verbose_name='Fwd 1Y %',
        orderable=True,
        #visible=False,
    )


    # Additional metrics
    fwd_volatility_1m = tables.Column(
        accessor='fwd_volatility_1m',
        verbose_name='Volatility 1M',
        orderable=True,
        #visible=False,
    )

    fwd_volatility_3m = tables.Column(
        accessor='fwd_volatility_3m',
        verbose_name='Volatility 3M',
        orderable=True,
        #visible=False,
    )

    fwd_volatility_6m = tables.Column(
        accessor='fwd_volatility_6m',
        verbose_name='Volatility 6M',
        orderable=True,
        #visible=False,
    )

    fwd_sharpe_ratio = tables.Column(
        accessor='fwd_sharpe_ratio',
        verbose_name='Sharpe Ratio',
        orderable=True,
        #visible=False,
    )

    fwd_max_drawdown = tables.Column(
        accessor='fwd_max_drawdown',
        verbose_name='Max Drawdown',
        orderable=True,
        #visible=False,
    )

    fwd_drawdown_duration = tables.Column(
        accessor='fwd_drawdown_duration',
        verbose_name='Drawdown Duration',
        orderable=True,
        #visible=False,
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
        try:
            float_val = float(value)
            if float_val >= 1000:
                return format_html('${:.1f}T', float_val / 1000)
            return format_html('${:.1f}B', float_val)
        except:
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
        try:
            return format_html('${:.2f}', float(value))
        except:
            return str(value)

    def render_change_1d(self, value, record):
        if value is None:
            return '-'
        try:
            color = 'text-success' if float(value) > 0 else 'text-danger'
            return format_html('<span class="{}">{:.2f}%</span>', color, float(value))
        except:
            return str(value)

    def render_change_1w(self, value, record):
        if value is None:
            return '-'
        try:
            color = 'text-success' if float(value) > 0 else 'text-danger'
            return format_html('<span class="{}">{:.2f}%</span>', color, float(value))
        except:
            return str(value)

    def render_change_2w(self, value, record):
        if value is None:
            return '-'
        try:
            color = 'text-success' if float(value) > 0 else 'text-danger'
            return format_html('<span class="{}">{:.2f}%</span>', color, float(value))
        except:
            return str(value)

    def render_change_1m(self, value, record):
        if value is None:
            return '-'
        try:
            color = 'text-success' if float(value) > 0 else 'text-danger'
            return format_html('<span class="{}">{:.2f}%</span>', color, float(value))
        except:
            return str(value)

    def render_change_3m(self, value, record):
        if value is None:
            return '-'
        try:
            color = 'text-success' if float(value) > 0 else 'text-danger'
            return format_html('<span class="{}">{:.2f}%</span>', color, float(value))
        except:
            return str(value)

    def render_change_6m(self, value, record):
        if value is None:
            return '-'
        try:
            color = 'text-success' if float(value) > 0 else 'text-danger'
            return format_html('<span class="{}">{:.2f}%</span>', color, float(value))
        except:
            return str(value)

    def render_change_1y(self, value, record):
        if value is None:
            return '-'
        try:
            color = 'text-success' if float(value) > 0 else 'text-danger'
            return format_html('<span class="{}">{:.2f}%</span>', color, float(value))
        except:
            return str(value)

    def render_fwd_change_1d(self, value, record):
        if value is None:
            return '-'
        try:
            return format_html('<span class="text-success">{:.2f}%</span>', float(value))
        except:
            return str(value)

    def render_fwd_change_1w(self, value, record):
        if value is None:
            return '-'
        try:
            return format_html('<span class="text-success">{:.2f}%</span>', float(value))
        except:
            return str(value)

    def render_fwd_change_2w(self, value, record):
        if value is None:
            return '-'
        try:
            return format_html('<span class="text-success">{:.2f}%</span>', float(value))
        except:
            return str(value)

    def render_fwd_change_1m(self, value, record):
        if value is None:
            return '-'
        try:
            return format_html('<span class="text-success">{:.2f}%</span>', float(value))
        except:
            return str(value)

    def render_fwd_change_3m(self, value, record):
        if value is None:
            return '-'
        try:
            return format_html('<span class="text-success">{:.2f}%</span>', float(value))
        except:
            return str(value)

    def render_fwd_change_6m(self, value, record):
        if value is None:
            return '-'
        try:
            return format_html('<span class="text-success">{:.2f}%</span>', float(value))
        except:
            return str(value)

    def render_fwd_change_1y(self, value, record):
        if value is None:
            return '-'
        try:
            return format_html('<span class="text-success">{:.2f}%</span>', float(value))
        except:
            return str(value)


    def render_fwd_volatility_1m(self, value, record):
        if value is None:
            return '-'
        try:
            return format_html('{:.2f}%', float(value))
        except:
            return str(value)

    def render_fwd_volatility_3m(self, value, record):
        if value is None:
            return '-'
        try:
            return format_html('{:.2f}%', float(value))
        except:
            return str(value)

    def render_fwd_volatility_6m(self, value, record):
        if value is None:
            return '-'
        try:
            return format_html('{:.2f}%', float(value))
        except:
            return str(value)

    def render_fwd_sharpe_ratio(self, value, record):
        if value is None:
            return '-'
        try:
            color = 'text-success' if float(value) > 0 else 'text-danger'
            return format_html('<span class="{}">{:.2f}</span>', color, float(value))
        except:
            return str(value)

    def render_fwd_max_drawdown(self, value, record):
        if value is None:
            return '-'
        try:
            return format_html('<span class="text-danger">{:.2f}%</span>', float(value))
        except:
            return str(value)

    def render_fwd_drawdown_duration(self, value, record):
        if value is None:
            return '-'
        try:
            return str(int(value))
        except:
            return str(value)

    class Meta:
        model = PrecomputedMetrics
        template_name = "django_tables2/bootstrap5.html"
        fields = [
            'ticker', 'name', 'market_cap', 'sector', 'industry',
            'current_price',
            'change_1d', 'change_1w', 'change_2w',
            'change_1m', 'change_3m', 'change_6m', 'change_1y',
            'fwd_change_1d', 'fwd_change_1w', 'fwd_change_2w',
            'fwd_change_1m', 'fwd_change_3m', 'fwd_change_6m', 'fwd_change_1y',
            'fwd_volatility_1m', 'fwd_volatility_3m', 'fwd_volatility_6m',
            'fwd_sharpe_ratio', 'fwd_max_drawdown', 'fwd_drawdown_duration',
        ]
        attrs = {
            'class': 'table table-striped table-hover table-sm',
            'thead': {'class': 'table-dark'},
            'id': 'market-data-table',
        }
        order_by = 'ticker'
        per_page = 25
