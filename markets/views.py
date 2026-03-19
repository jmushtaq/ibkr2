from django.shortcuts import render
from django.core.cache import cache
from django.db.models import Q, F
from django.views.generic import ListView
from django_tables2 import SingleTableView, RequestConfig
from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin
from django.contrib.postgres.search import SearchVector
from django.http import JsonResponse
import logging

from .models import Symbol, PrecomputedMetrics, OHLCVData
from .tables import SymbolMetricsTable
from .filters import SymbolMetricsFilter
from ibkr_project.settings import CACHE_TTL

logger = logging.getLogger(__name__)


class SymbolMetricsListView(SingleTableMixin, FilterView):
    model = PrecomputedMetrics
    table_class = SymbolMetricsTable
    template_name = 'markets/market_list.html'
    filterset_class = SymbolMetricsFilter
    paginate_by = 50

    def get_queryset(self):
        """Optimize queryset with select_related"""
        try:
            queryset = PrecomputedMetrics.objects.filter(
                frequency='1D'
            ).select_related(
                'symbol__sector',
                'symbol__industry'
            ).order_by('symbol__ticker')

            # Get the latest date by default
            latest_date = PrecomputedMetrics.objects.filter(
                frequency='1D'
            ).order_by('-as_of_date').values_list('as_of_date', flat=True).first()

            if latest_date:
                queryset = queryset.filter(as_of_date=latest_date)

            return queryset
        except Exception as e:
            logger.error(f"Error in get_queryset: {str(e)}")
            return PrecomputedMetrics.objects.none()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Market Dashboard'
        context['year_range'] = range(2020, 2027)  # Adjust as needed

        # Add filter form to context with current values
        if self.filterset:
            context['filter_form'] = self.filterset.form

        return context

    def get(self, request, *args, **kwargs):
        try:
            response = super().get(request, *args, **kwargs)

            # Check if this is an HTMX request for partial updates
            if request.headers.get('HX-Request'):
                return render(request, 'markets/partials/table_rows.html', {
                    'table': self.get_table(),
                    'filter': self.filterset,
                })

            return response
        except Exception as e:
            logger.error(f"Error in SymbolMetricsListView.get: {str(e)}")
            # Return a basic error response
            return render(request, 'markets/market_list.html', {
                'error': f"An error occurred: {str(e)}",
                'title': 'Market Dashboard',
                'year_range': range(2020, 2027),
            })


class ChartDataView(FilterView):
    """API endpoint for chart data"""

    def get(self, request, *args, **kwargs):
        ticker = request.GET.get('ticker')
        frequency = request.GET.get('frequency', '1D')
        start_year = int(request.GET.get('start_year', 2020))
        end_year = int(request.GET.get('end_year', 2026))

        if not ticker:
            return JsonResponse({'error': 'Ticker required'}, status=400)

        # Try cache first
        cache_key = f"chart_data_{ticker}_{frequency}_{start_year}_{end_year}"
        cached_data = cache.get(cache_key)
        if cached_data:
            return JsonResponse(cached_data)

        try:
            symbol = Symbol.objects.get(ticker=ticker)

            # Get OHLCV data for the year range
            ohlcv_data = OHLCVData.objects.filter(
                symbol=symbol,
                frequency=frequency,
                year__gte=start_year,
                year__lte=end_year
            ).order_by('year')

            if not ohlcv_data:
                return JsonResponse({'error': 'No data found'}, status=404)

            # Combine data from multiple years
            import pandas as pd
            dfs = []
            for data in ohlcv_data:
                df = data.get_data_as_dataframe()
                dfs.append(df)

            if dfs:
                combined_df = pd.concat(dfs)
                combined_df.sort_index(inplace=True)

                # Convert to JSON serializable format
                result = {
                    'dates': combined_df.index.strftime('%Y-%m-%d').tolist(),
                    'open': combined_df['open'].tolist(),
                    'high': combined_df['high'].tolist(),
                    'low': combined_df['low'].tolist(),
                    'close': combined_df['close'].tolist(),
                    'volume': combined_df['volume'].tolist(),
                }

                # Cache for 15 minutes
                cache.set(cache_key, result, CACHE_TTL)

                return JsonResponse(result)

        except Symbol.DoesNotExist:
            return JsonResponse({'error': 'Symbol not found'}, status=404)
        except Exception as e:
            logger.error(f"Error fetching chart data: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

        return JsonResponse({'error': 'No data available'}, status=404)

