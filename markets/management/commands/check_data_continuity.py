"""
Management command to check data continuity for symbols.
Helps identify gaps in historical data.

Usage:
    python manage.py check_data_continuity --frequency 1D
    python manage.py check_data_continuity --frequency 1D --symbol AAPL
"""

from django.core.management.base import BaseCommand
from markets.models import Symbol, OHLCVData
import pandas as pd
from datetime import timedelta

class Command(BaseCommand):
    help = 'Check data continuity for symbols'

    def add_arguments(self, parser):
        parser.add_argument('--frequency', type=str, default='1D', help='Frequency to check')
        parser.add_argument('--symbol', type=str, help='Specific symbol to check')
        parser.add_argument('--gap-threshold', type=int, default=5,
                          help='Max allowed gap in days before reporting (default: 5)')

    def handle(self, *args, **options):
        frequency = options.get('frequency', '1D')
        symbol_filter = options.get('symbol')
        gap_threshold = options.get('gap_threshold', 5)

        # Get symbols to check
        if symbol_filter:
            symbols = Symbol.objects.filter(ticker=symbol_filter)
        else:
            symbols = Symbol.objects.filter(is_active=True).order_by('ticker')[:10]  # Limit for demo

        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"DATA CONTINUITY CHECK FOR {frequency}")
        self.stdout.write(f"{'='*60}\n")

        for symbol in symbols:
            self.stdout.write(f"\nChecking {symbol.ticker}...")

            # Get all OHLCV data for this symbol
            ohlcv_data = OHLCVData.objects.filter(
                symbol=symbol,
                frequency=frequency
            ).order_by('year')

            if not ohlcv_data.exists():
                self.stdout.write(self.style.WARNING(f"  No data found"))
                continue

            # Concatenate all data
            dfs = []
            for data in ohlcv_data:
                df = data.get_data_as_dataframe()
                if df is not None and not df.empty:
                    dfs.append(df)

            if not dfs:
                self.stdout.write(self.style.WARNING(f"  No valid data"))
                continue

            combined_df = pd.concat(dfs)
            combined_df.sort_index(inplace=True)
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

            # Check continuity
            dates = combined_df.index
            date_range = pd.date_range(start=dates[0], end=dates[-1], freq='D')

            missing_dates = []
            for expected_date in date_range:
                if expected_date not in dates:
                    missing_dates.append(expected_date)

            # Find gaps
            gaps = []
            current_gap = []

            for date in sorted(missing_dates):
                if not current_gap:
                    current_gap = [date]
                elif (date - current_gap[-1]).days == 1:
                    current_gap.append(date)
                else:
                    if len(current_gap) >= gap_threshold:
                        gaps.append((current_gap[0], current_gap[-1], len(current_gap)))
                    current_gap = [date]

            if current_gap and len(current_gap) >= gap_threshold:
                gaps.append((current_gap[0], current_gap[-1], len(current_gap)))

            # Report
            self.stdout.write(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
            self.stdout.write(f"  Total days in data: {len(dates)}")
            self.stdout.write(f"  Expected days: {len(date_range)}")
            self.stdout.write(f"  Missing days: {len(missing_dates)}")

            if gaps:
                self.stdout.write(self.style.WARNING(f"  Significant gaps found:"))
                for gap_start, gap_end, length in gaps:
                    self.stdout.write(f"    - {gap_start.date()} to {gap_end.date()} ({length} days)")
            else:
                self.stdout.write(self.style.SUCCESS(f"  No significant gaps found"))
