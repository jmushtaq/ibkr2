from rest_framework import serializers
from .models import Symbol, PrecomputedMetrics

class SymbolMetricsSerializer(serializers.ModelSerializer):
    ticker = serializers.CharField(source='symbol.ticker')
    name = serializers.CharField(source='symbol.name')
    market_cap = serializers.DecimalField(source='symbol.market_cap', max_digits=20, decimal_places=2)
    sector = serializers.CharField(source='symbol.sector.name', default='')
    industry = serializers.CharField(source='symbol.industry.name', default='')

    class Meta:
        model = PrecomputedMetrics
        fields = [
            'ticker', 'name', 'market_cap', 'sector', 'industry',
            'current_price', 'change_1d', 'change_1w', 'change_2w',
            'change_1m', 'change_3m', 'change_6m', 'change_1y',
            'as_of_date'
        ]
