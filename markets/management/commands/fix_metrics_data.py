from django.core.management.base import BaseCommand
from markets.models import PrecomputedMetrics, Symbol, OHLCVData
from django.utils import timezone
from datetime import timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Fix missing PrecomputedMetrics data'

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, help='Fix specific symbol')
        parser.add_argument('--all', action='store_true', help='Fix all symbols')
        parser.add_argument('--date', type=str, help='Fix for specific date (YYYY-MM-DD)')

    def handle(self, *args, **options):
        symbol_filter = options.get('symbol')
        fix_all = options.get('all', False)
        date_str = options.get('date')

        if not (symbol_filter or fix_all):
            self.stdout.write(self.style.ERROR("Please specify --symbol or --all"))
            return

        # Get symbols to process
        if fix_all:
            symbols = Symbol.objects.filter(is_active=True)
        else:
            symbols = Symbol.objects.filter(ticker=symbol_filter)

        if date_str:
            target_date = timezone.datetime.strptime(date_str, '%Y-%m-%d').date()
        else:
            target_date = None

        self.stdout.write(f"Processing {symbols.count()} symbols...")

        fixed_count = 0
        error_count = 0

        for symbol in symbols:
            try:
                self.stdout.write(f"\nProcessing {symbol.ticker}...")

                # Get OHLCV data for this symbol
                ohlcv_data = OHLCVData.objects.filter(
                    symbol=symbol,
                    frequency='1D'
                ).order_by('year')

                if not ohlcv_data:
                    self.stdout.write(self.style.WARNING(f"  No OHLCV data for {symbol.ticker}"))
                    continue

                # Combine data from all years
                dfs = []
                for data in ohlcv_data:
                    df = data.get_data_as_dataframe()
                    if df is not None and not df.empty:
                        dfs.append(df)

                if not dfs:
                    self.stdout.write(self.style.WARNING(f"  No valid dataframe for {symbol.ticker}"))
                    continue

                combined_df = pd.concat(dfs)
                combined_df.sort_index(inplace=True)

                # If specific date provided, use that, otherwise use latest date
                if target_date:
                    # Convert target_date to datetime for comparison
                    target_datetime = pd.Timestamp(target_date)
                    if target_datetime in combined_df.index:
                        latest_date = target_datetime
                    else:
                        self.stdout.write(self.style.WARNING(f"  Date {target_date} not found in data"))
                        continue
                else:
                    latest_date = combined_df.index[-1]

                latest_price = float(combined_df.loc[latest_date, 'close'])

                # Calculate changes for different periods
                changes = self.calculate_changes(combined_df, latest_date)

                # Update or create metrics
                metrics, created = PrecomputedMetrics.objects.update_or_create(
                    symbol=symbol,
                    frequency='1D',
                    as_of_date=latest_date.date(),
                    defaults={
                        'current_price': latest_price,
                        'change_1d': changes.get('change_1d'),
                        'change_1w': changes.get('change_1w'),
                        'change_2w': changes.get('change_2w'),
                        'change_1m': changes.get('change_1m'),
                        'change_3m': changes.get('change_3m'),
                        'change_6m': changes.get('change_6m'),
                        'change_1y': changes.get('change_1y'),
                    }
                )

                action = "Created" if created else "Updated"
                self.stdout.write(self.style.SUCCESS(
                    f"  {action} metrics for {symbol.ticker} as of {latest_date.date()}"
                ))
                self.stdout.write(f"    Price: ${latest_price:.2f}")
                self.stdout.write(f"    1D: {changes.get('change_1d', 'N/A')}%")
                self.stdout.write(f"    1W: {changes.get('change_1w', 'N/A')}%")
                self.stdout.write(f"    1M: {changes.get('change_1m', 'N/A')}%")
                self.stdout.write(f"    1Y: {changes.get('change_1y', 'N/A')}%")

                fixed_count += 1

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  Error processing {symbol.ticker}: {str(e)}"))
                logger.error(f"Error fixing metrics for {symbol.ticker}: {str(e)}", exc_info=True)
                error_count += 1

        self.stdout.write(self.style.SUCCESS(
            f"\nDone! Fixed: {fixed_count}, Errors: {error_count}"
        ))

    def calculate_changes(self, df, latest_date):
        """Calculate percentage changes for different periods"""
        changes = {}

        # Periods in days
        periods = {
            'change_1d': 1,
            'change_1w': 7,
            'change_2w': 14,
            'change_1m': 30,
            'change_3m': 90,
            'change_6m': 180,
            'change_1y': 365,
        }

        latest_price = float(df.loc[latest_date, 'close'])

        for field, days in periods.items():
            try:
                target_date = latest_date - pd.Timedelta(days=days)

                # Find the closest date that's not after target_date
                available_dates = df.index[df.index <= target_date]
                if len(available_dates) > 0:
                    closest_date = available_dates[-1]
                    past_price = float(df.loc[closest_date, 'close'])

                    if past_price and past_price > 0:
                        pct_change = ((latest_price - past_price) / past_price) * 100
                        changes[field] = round(pct_change, 2)
                    else:
                        changes[field] = None
                else:
                    changes[field] = None

            except Exception as e:
                logger.warning(f"Error calculating {field}: {str(e)}")
                changes[field] = None

        return changes

