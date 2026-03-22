import django_filters
from django import forms
from django.db.models import Q
from .models import PrecomputedMetrics, Symbol, Sector, Industry


class SymbolMetricsFilter(django_filters.FilterSet):
    # Text search
    search = django_filters.CharFilter(
        method='filter_search',
        label='Search',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Search by ticker or company...'})
    )

    # Sector filter
    sector = django_filters.ModelChoiceFilter(
        field_name='symbol__sector',
        queryset=Sector.objects.all(),
        empty_label='All Sectors',
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    # Industry filter
    industry = django_filters.ModelChoiceFilter(
        field_name='symbol__industry',
        queryset=Industry.objects.all(),
        empty_label='All Industries',
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    # Year range filter
    year_from = django_filters.NumberFilter(
        field_name='as_of_date__year',
        lookup_expr='gte',
        label='From Year',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'From'})
    )

    year_to = django_filters.NumberFilter(
        field_name='as_of_date__year',
        lookup_expr='lte',
        label='To Year',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'To'})
    )

    # Historical Percentage change filters
    change_1d_min = django_filters.NumberFilter(
        field_name='change_1d',
        lookup_expr='gte',
        label='1D % Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    change_1d_max = django_filters.NumberFilter(
        field_name='change_1d',
        lookup_expr='lte',
        label='1D % Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    change_1w_min = django_filters.NumberFilter(
        field_name='change_1w',
        lookup_expr='gte',
        label='1W % Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    change_1w_max = django_filters.NumberFilter(
        field_name='change_1w',
        lookup_expr='lte',
        label='1W % Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    change_1m_min = django_filters.NumberFilter(
        field_name='change_1m',
        lookup_expr='gte',
        label='1M % Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    change_1m_max = django_filters.NumberFilter(
        field_name='change_1m',
        lookup_expr='lte',
        label='1M % Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    change_1y_min = django_filters.NumberFilter(
        field_name='change_1y',
        lookup_expr='gte',
        label='1Y % Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    change_1y_max = django_filters.NumberFilter(
        field_name='change_1y',
        lookup_expr='lte',
        label='1Y % Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    # Forward-looking filters
    fwd_change_1d_min = django_filters.NumberFilter(
        field_name='fwd_change_1d',
        lookup_expr='gte',
        label='Fwd 1D % Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    fwd_change_1d_max = django_filters.NumberFilter(
        field_name='fwd_change_1d',
        lookup_expr='lte',
        label='Fwd 1D % Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    fwd_change_1w_min = django_filters.NumberFilter(
        field_name='fwd_change_1w',
        lookup_expr='gte',
        label='Fwd 1W % Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    fwd_change_1w_max = django_filters.NumberFilter(
        field_name='fwd_change_1w',
        lookup_expr='lte',
        label='Fwd 1W % Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    fwd_change_2w_min = django_filters.NumberFilter(
        field_name='fwd_change_2w',
        lookup_expr='gte',
        label='Fwd 2W % Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    fwd_change_2w_max = django_filters.NumberFilter(
        field_name='fwd_change_2w',
        lookup_expr='lte',
        label='Fwd 2W % Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    fwd_change_1m_min = django_filters.NumberFilter(
        field_name='fwd_change_1m',
        lookup_expr='gte',
        label='Fwd 1M % Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    fwd_change_1m_max = django_filters.NumberFilter(
        field_name='fwd_change_1m',
        lookup_expr='lte',
        label='Fwd 1M % Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    fwd_change_3m_min = django_filters.NumberFilter(
        field_name='fwd_change_3m',
        lookup_expr='gte',
        label='Fwd 3M % Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    fwd_change_3m_max = django_filters.NumberFilter(
        field_name='fwd_change_3m',
        lookup_expr='lte',
        label='Fwd 3M % Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    fwd_change_6m_min = django_filters.NumberFilter(
        field_name='fwd_change_6m',
        lookup_expr='gte',
        label='Fwd 6M % Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    fwd_change_6m_max = django_filters.NumberFilter(
        field_name='fwd_change_6m',
        lookup_expr='lte',
        label='Fwd 6M % Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    fwd_change_1y_min = django_filters.NumberFilter(
        field_name='fwd_change_1y',
        lookup_expr='gte',
        label='Fwd 1Y % Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    fwd_change_1y_max = django_filters.NumberFilter(
        field_name='fwd_change_1y',
        lookup_expr='lte',
        label='Fwd 1Y % Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    # Risk metrics filters
    fwd_volatility_1m_min = django_filters.NumberFilter(
        field_name='fwd_volatility_1m',
        lookup_expr='gte',
        label='Volatility 1M Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    fwd_volatility_1m_max = django_filters.NumberFilter(
        field_name='fwd_volatility_1m',
        lookup_expr='lte',
        label='Volatility 1M Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    fwd_volatility_3m_min = django_filters.NumberFilter(
        field_name='fwd_volatility_3m',
        lookup_expr='gte',
        label='Volatility 3M Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    fwd_volatility_3m_max = django_filters.NumberFilter(
        field_name='fwd_volatility_3m',
        lookup_expr='lte',
        label='Volatility 3M Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    fwd_volatility_6m_min = django_filters.NumberFilter(
        field_name='fwd_volatility_6m',
        lookup_expr='gte',
        label='Volatility 6M Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    fwd_volatility_6m_max = django_filters.NumberFilter(
        field_name='fwd_volatility_6m',
        lookup_expr='lte',
        label='Volatility 6M Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    fwd_sharpe_ratio_min = django_filters.NumberFilter(
        field_name='fwd_sharpe_ratio',
        lookup_expr='gte',
        label='Sharpe Ratio Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min'})
    )

    fwd_sharpe_ratio_max = django_filters.NumberFilter(
        field_name='fwd_sharpe_ratio',
        lookup_expr='lte',
        label='Sharpe Ratio Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max'})
    )

    fwd_max_drawdown_min = django_filters.NumberFilter(
        field_name='fwd_max_drawdown',
        lookup_expr='gte',
        label='Max Drawdown Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min %'})
    )

    fwd_max_drawdown_max = django_filters.NumberFilter(
        field_name='fwd_max_drawdown',
        lookup_expr='lte',
        label='Max Drawdown Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max %'})
    )

    fwd_drawdown_duration_min = django_filters.NumberFilter(
        field_name='fwd_drawdown_duration',
        lookup_expr='gte',
        label='Drawdown Duration Min (days)',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min days'})
    )

    fwd_drawdown_duration_max = django_filters.NumberFilter(
        field_name='fwd_drawdown_duration',
        lookup_expr='lte',
        label='Drawdown Duration Max (days)',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max days'})
    )

    # Market cap filter
    market_cap_min = django_filters.NumberFilter(
        field_name='symbol__market_cap',
        lookup_expr='gte',
        label='Market Cap Min (B)',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min B$'})
    )

    market_cap_max = django_filters.NumberFilter(
        field_name='symbol__market_cap',
        lookup_expr='lte',
        label='Market Cap Max (B)',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max B$'})
    )

    # Price filter
    price_min = django_filters.NumberFilter(
        field_name='current_price',
        lookup_expr='gte',
        label='Price Min',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Min $'})
    )

    price_max = django_filters.NumberFilter(
        field_name='current_price',
        lookup_expr='lte',
        label='Price Max',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max $'})
    )

    class Meta:
        model = PrecomputedMetrics
        fields = [
            'search', 'sector', 'industry', 'year_from', 'year_to',
            'change_1d_min', 'change_1d_max', 'change_1w_min', 'change_1w_max',
            'change_1m_min', 'change_1m_max', 'change_1y_min', 'change_1y_max',
            'fwd_change_1d_min', 'fwd_change_1d_max', 'fwd_change_1w_min', 'fwd_change_1w_max',
            'fwd_change_2w_min', 'fwd_change_2w_max', 'fwd_change_1m_min', 'fwd_change_1m_max',
            'fwd_change_3m_min', 'fwd_change_3m_max', 'fwd_change_6m_min', 'fwd_change_6m_max',
            'fwd_change_1y_min', 'fwd_change_1y_max',
            'fwd_volatility_1m_min', 'fwd_volatility_1m_max', 'fwd_volatility_3m_min', 'fwd_volatility_3m_max',
            'fwd_volatility_6m_min', 'fwd_volatility_6m_max',
            'fwd_sharpe_ratio_min', 'fwd_sharpe_ratio_max',
            'fwd_max_drawdown_min', 'fwd_max_drawdown_max',
            'fwd_drawdown_duration_min', 'fwd_drawdown_duration_max',
            'market_cap_min', 'market_cap_max', 'price_min', 'price_max'
        ]

    def filter_search(self, queryset, name, value):
        """Search across ticker and company name"""
        return queryset.filter(
            Q(symbol__ticker__icontains=value) |
            Q(symbol__name__icontains=value)
        )


class TechnicalIndicatorFilter(django_filters.FilterSet):
    rsi_14_min = django_filters.NumberFilter(
        field_name='technical_indicators__rsi_14',
        lookup_expr='gte'
    )
    rsi_14_max = django_filters.NumberFilter(
        field_name='technical_indicators__rsi_14',
        lookup_expr='lte'
    )
    sma_50_above_200 = django_filters.BooleanFilter(
        method='filter_sma_crossover'
    )

    def filter_sma_crossover(self, queryset, name, value):
        if value:
            return queryset.filter(
                technical_indicators__sma_50__gt=models.F('technical_indicators__sma_200')
            )
        return queryset

