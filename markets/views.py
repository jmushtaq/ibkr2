from django.shortcuts import render
from django.core.cache import cache
from django.db.models import Q, F
from django.views.generic import ListView
from django_tables2 import SingleTableView, RequestConfig
from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin
from django.contrib.postgres.search import SearchVector
from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest
from django.views.generic import TemplateView
import csv
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
        """Get queryset with data for selected date"""
        try:
            # Get the selected date from request, or use latest date
            selected_date = self.request.GET.get('as_of_date')

            if selected_date:
                try:
                    # Parse the selected date
                    from datetime import datetime
                    as_of_date = datetime.strptime(selected_date, '%Y-%m-%d').date()
                except ValueError:
                    as_of_date = None
            else:
                as_of_date = None

            if not as_of_date:
                # Get the latest date if no date selected
                as_of_date = PrecomputedMetrics.objects.filter(
                    frequency='1D'
                ).order_by('-as_of_date').values_list('as_of_date', flat=True).first()

            if not as_of_date:
                return PrecomputedMetrics.objects.none()

            # Get records for the selected date with related data
            queryset = PrecomputedMetrics.objects.filter(
                frequency='1D',
                as_of_date=as_of_date
            ).select_related(
                'symbol__sector',
                'symbol__industry'
            ).prefetch_related(
                'technical_indicators'  # Now this works because TechnicalIndicators has ForeignKey to PrecomputedMetrics
            ).order_by('symbol__ticker')

            return queryset

        except Exception as e:
            print(f"Error in get_queryset: {str(e)}")
            import traceback
            traceback.print_exc()
            return PrecomputedMetrics.objects.none()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Market Dashboard'
        context['year_range'] = range(2020, 2027)

        # Get all available dates for the date picker
        available_dates = PrecomputedMetrics.objects.filter(
            frequency='1D'
        ).values_list('as_of_date', flat=True).distinct().order_by('-as_of_date')

        context['available_dates'] = available_dates

        # Get current selected date
        selected_date = self.request.GET.get('as_of_date')
        if selected_date:
            context['selected_date'] = selected_date
        else:
            # Default to latest date
            latest_date = available_dates.first()
            if latest_date:
                context['selected_date'] = latest_date.strftime('%Y-%m-%d')

        # Add debug info
        if hasattr(self, 'object_list') and self.object_list.exists():
            context['debug_record_count'] = self.object_list.count()
            first = self.object_list.first()
            context['debug_first'] = {
                'ticker': first.symbol.ticker,
                'price': first.current_price,
                'change_1d': first.change_1d,
            }

        return context


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
                if df is not None and not df.empty:
                    dfs.append(df)

            if dfs:
                combined_df = pd.concat(dfs)
                combined_df.sort_index(inplace=True)

                # Debug: print first few dates
                logger.info(f"First few dates for {ticker}: {combined_df.index[:5].tolist()}")

                # Convert dates to string format that JavaScript can parse correctly
                # Ensure we have valid datetime objects
                dates = []
                for date in combined_df.index:
                    if pd.isna(date):
                        dates.append(None)
                    else:
                        # Convert to datetime and then to ISO format
                        if hasattr(date, 'strftime'):
                            dates.append(date.strftime('%Y-%m-%dT%H:%M:%S'))
                        else:
                            # Try to convert to datetime
                            try:
                                dt = pd.to_datetime(date)
                                dates.append(dt.strftime('%Y-%m-%dT%H:%M:%S'))
                            except:
                                dates.append('1970-01-01T00:00:00')

                # Ensure all numeric values are proper Python floats
                result = {
                    'dates': dates,
                    'open': [float(x) if pd.notna(x) else None for x in combined_df['open'].tolist()],
                    'high': [float(x) if pd.notna(x) else None for x in combined_df['high'].tolist()],
                    'low': [float(x) if pd.notna(x) else None for x in combined_df['low'].tolist()],
                    'close': [float(x) if pd.notna(x) else None for x in combined_df['close'].tolist()],
                    'volume': [float(x) if pd.notna(x) else None for x in combined_df['volume'].tolist()],
                }

                # Cache for 15 minutes
                cache.set(cache_key, result, CACHE_TTL)

                return JsonResponse(result)

        except Symbol.DoesNotExist:
            return JsonResponse({'error': 'Symbol not found'}, status=404)
        except Exception as e:
            logger.error(f"Error fetching chart data: {str(e)}", exc_info=True)
            return JsonResponse({'error': str(e)}, status=500)

        return JsonResponse({'error': 'No data available'}, status=404)


class ExportDataView(SingleTableMixin, FilterView):
    """Export filtered data to CSV with column selection"""

    def get(self, request, *args, **kwargs):
        # Get filtered queryset
        self.filterset = self.get_filterset()
        self.object_list = self.filterset.qs

        # Get selected columns from session or use defaults
        export_columns = request.session.get('export_columns',
            getattr(settings, 'MARKET_DASHBOARD_DEFAULT_COLUMNS', [
                'ticker', 'name', 'market_cap', 'sector', 'industry',
                'current_price', 'change_1d', 'change_1w', 'change_1m', 'change_1y'
            ]))

        # Create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="market_data.csv"'

        writer = csv.writer(response)

        # Define column headers mapping
        column_headers = {
            'ticker': 'Symbol',
            'name': 'Company Name',
            'market_cap': 'Market Cap (B)',
            'sector': 'Sector',
            'industry': 'Industry',
            'current_price': 'Price',
            'change_1d': '1D %',
            'change_1w': '1W %',
            'change_2w': '2W %',
            'change_1m': '1M %',
            'change_3m': '3M %',
            'change_6m': '6M %',
            'change_1y': '1Y %',
            'volume': 'Volume',
            'avg_volume_20d': 'Avg Vol (20D)',
            'pe_ratio': 'P/E',
            'dividend_yield': 'Div Yield %',
            'week_52_high': '52W High',
            'week_52_low': '52W Low',
            'relative_volume': 'Rel Volume',
            'atr': 'ATR',
            'rsi': 'RSI',
        }

        # Write header with selected columns
        header = [column_headers[col] for col in export_columns if col in column_headers]
        writer.writerow(header)

        # Write data for selected columns only
        for obj in self.object_list.select_related('symbol__sector', 'symbol__industry'):
            row = []
            for col in export_columns:
                if col == 'ticker':
                    row.append(obj.symbol.ticker)
                elif col == 'name':
                    row.append(obj.symbol.name)
                elif col == 'market_cap':
                    row.append(obj.symbol.market_cap)
                elif col == 'sector':
                    row.append(obj.symbol.sector.name if obj.symbol.sector else '')
                elif col == 'industry':
                    row.append(obj.symbol.industry.name if obj.symbol.industry else '')
                elif col == 'current_price':
                    row.append(obj.current_price)
                elif hasattr(obj, col):
                    value = getattr(obj, col)
                    row.append(value)
                else:
                    row.append('')
            writer.writerow(row)

        return response


class ChartPageView(FilterView):
    """View for the chart page that opens in a new tab"""
    template_name = 'markets/chart_page.html'

    def get(self, request, *args, **kwargs):
        ticker = request.GET.get('ticker')
        if not ticker:
            return HttpResponseBadRequest("Ticker required")

        context = {
            'ticker': ticker,
            'year_range': range(2020, 2027),
        }
        return render(request, self.template_name, context)


class SymbolListView(FilterView):
    """API endpoint to get all symbols for the dropdown"""

    def get(self, request, *args, **kwargs):
        symbols = Symbol.objects.filter(is_active=True).values_list('ticker', flat=True).order_by('ticker')
        return JsonResponse({'symbols': list(symbols)})


class IndicatorDataView(FilterView):
    """API endpoint for technical indicator data"""

    def get(self, request, *args, **kwargs):
        ticker = request.GET.get('ticker')
        frequency = request.GET.get('frequency', '1D')
        start_year = int(request.GET.get('start_year', 2020))
        end_year = int(request.GET.get('end_year', 2026))
        indicators = request.GET.getlist('indicators[]')

        if not ticker:
            return JsonResponse({'error': 'Ticker required'}, status=400)

        if not indicators:
            return JsonResponse({})

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
                if df is not None and not df.empty:
                    dfs.append(df)

            if dfs:
                combined_df = pd.concat(dfs)
                combined_df.sort_index(inplace=True)

                # Calculate indicators
                from .indicators import TechnicalIndicators
                indicator_results = TechnicalIndicators.calculate_indicators(
                    combined_df, indicators
                )

                # Prepare dates for response
                dates = combined_df.index.strftime('%Y-%m-%dT%H:%M:%S').tolist()

                result = {
                    'dates': dates,
                    'indicators': indicator_results
                }

                return JsonResponse(result)

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}", exc_info=True)
            return JsonResponse({'error': str(e)}, status=500)

        return JsonResponse({'error': 'No data available'}, status=404)


class AvailableIndicatorsView(FilterView):
    """API endpoint to get available technical indicators"""

    def get(self, request, *args, **kwargs):
        from .indicators import TechnicalIndicators
        indicators = TechnicalIndicators.get_available_indicators()
        return JsonResponse({'indicators': indicators})


class ChartPageView(FilterView):
    """View for the chart page that opens in a new tab"""
    template_name = 'markets/chart_tv.html'  # Changed from chart_page.html

    def get(self, request, *args, **kwargs):
        ticker = request.GET.get('ticker')
        if not ticker:
            return HttpResponseBadRequest("Ticker required")

        context = {
            'ticker': ticker,
            'year_range': range(2020, 2027),
        }
        return render(request, self.template_name, context)


class DebugDataView(FilterView):
    """Debug view to check raw data"""
    template_name = 'markets/debug_data.html'

    def get(self, request, *args, **kwargs):
        # Get latest metrics
        latest_date = PrecomputedMetrics.objects.filter(
            frequency='1D'
        ).order_by('-as_of_date').values_list('as_of_date', flat=True).first()

        metrics = PrecomputedMetrics.objects.filter(
            frequency='1D',
            as_of_date=latest_date
        ).select_related('symbol')[:50]

        context = {
            'metrics': metrics,
            'total_count': PrecomputedMetrics.objects.filter(frequency='1D').count(),
            'latest_date': latest_date,
        }
        return render(request, self.template_name, context)


def diagnostic_view(request):
    """Diagnostic endpoint to check data"""
    import json
    from django.core import serializers

    # Check PrecomputedMetrics
    metrics_count = PrecomputedMetrics.objects.filter(frequency='1D').count()
    dates = PrecomputedMetrics.objects.filter(frequency='1D').values_list('as_of_date', flat=True).distinct().order_by('-as_of_date')

    # Get a sample of records
    sample_records = []
    if dates:
        latest_date = dates.first()
        sample = PrecomputedMetrics.objects.filter(
            frequency='1D',
            as_of_date=latest_date
        ).select_related('symbol')[:5]

        for record in sample:
            sample_records.append({
                'ticker': record.symbol.ticker,
                'date': str(record.as_of_date),
                'price': record.current_price,
                'change_1d': record.change_1d,
                'change_1w': record.change_1w,
                'change_1m': record.change_1m,
                'change_1y': record.change_1y,
            })

    return JsonResponse({
        'metrics_count': metrics_count,
        'available_dates': [str(d) for d in dates[:10]],
        'sample_records': sample_records,
        'latest_date': str(dates.first()) if dates else None,
    })


# markets/views.py - Update SimpleMarketView

class SimpleMarketView(SingleTableMixin, FilterView):
    """Simplified view for testing"""
    model = PrecomputedMetrics
    table_class = SymbolMetricsTable
    template_name = 'markets/simple_market_list.html'  # Use a different template
    paginate_by = 50

    def get_queryset(self):
        # Get the latest date first
        latest_date = PrecomputedMetrics.objects.filter(
            frequency='1D'
        ).order_by('-as_of_date').values_list('as_of_date', flat=True).first()

        if not latest_date:
            return PrecomputedMetrics.objects.none()

        # Return full queryset
        return PrecomputedMetrics.objects.filter(
            frequency='1D',
            as_of_date=latest_date
        ).select_related(
            'symbol__sector',
            'symbol__industry'
        ).order_by('symbol__ticker')


class MinimalTestView(TemplateView):
    """Minimal test view to check data access"""
    template_name = 'markets/minimal_test.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Get some data directly
        latest_date = PrecomputedMetrics.objects.filter(
            frequency='1D'
        ).order_by('-as_of_date').values_list('as_of_date', flat=True).first()

        if latest_date:
            records = PrecomputedMetrics.objects.filter(
                frequency='1D',
                as_of_date=latest_date
            ).select_related('symbol')[:5]

            context['records'] = records
            context['record_count'] = records.count()
            context['latest_date'] = latest_date
        else:
            context['records'] = []
            context['record_count'] = 0
            context['latest_date'] = None

        # Get total count
        context['total_count'] = PrecomputedMetrics.objects.filter(frequency='1D').count()

        return context


def available_dates_view(request):
    """API endpoint to get available dates"""
    dates = PrecomputedMetrics.objects.filter(
        frequency='1D'
    ).values_list('as_of_date', flat=True).distinct().order_by('-as_of_date')

    return JsonResponse({
        'dates': [d.strftime('%Y-%m-%d') for d in dates]
    })

