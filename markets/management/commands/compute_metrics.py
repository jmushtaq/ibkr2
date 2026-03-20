"""
Management command to compute PrecomputedMetrics for all dates using concatenated historical data.
Run this after loading all OHLCV data.

Usage:
    python manage.py compute_metrics --frequency 1D
    python manage.py compute_metrics --frequency 1D --symbol AAPL
    python manage.py compute_metrics --frequency 1D --all-symbols
    python manage.py compute_metrics --frequency 1D --start-date 2020-01-01 --end-date 2026-12-31

    Shell to check data continuitybetween years (concat years):
        from markets.models import PrecomputedMetrics
        from datetime import date

        # Check metrics for a specific symbol and date
        symbol = 'AAPL'  # Replace with your symbol
        test_date = date(2026, 1, 30)

        metrics = PrecomputedMetrics.objects.filter(
            symbol__ticker=symbol,
            frequency='1D',
            as_of_date=test_date
        ).first()

        if metrics:
            print(f"Metrics for {symbol} on {test_date}:")
            print(f"  Price: ${metrics.current_price}")
            print(f"  1D: {metrics.change_1d}%")
            print(f"  1W: {metrics.change_1w}%")
            print(f"  1M: {metrics.change_1m}%")
            print(f"  6M: {metrics.change_6m}%")
            print(f"  1Y: {metrics.change_1y}%")
        else:
            print(f"No metrics found for {symbol} on {test_date}")

            # Check if we have OHLCV data for that date
            from markets.models import OHLCVData
            import pandas as pd

            ohlcv = OHLCVData.objects.filter(
                symbol__ticker=symbol,
                frequency='1D'
            ).order_by('year')

            dfs = []
            for data in ohlcv:
                df = data.get_data_as_dataframe()
                if df is not None and not df.empty:
                    dfs.append(df)

            if dfs:
                combined = pd.concat(dfs)
                combined.sort_index(inplace=True)

                test_timestamp = pd.Timestamp(test_date).tz_localize('UTC')
                if test_timestamp in combined.index:
                    print(f"OHLCV data exists for {test_date}")
                    price = combined.loc[test_timestamp, 'close']
                    print(f"  Price: ${price}")
                else:
                    print(f"No OHLCV data for {test_date}")
                    # Show available dates around that time
                    dates = combined.index
                    before = dates[dates <= test_timestamp][-5:] if any(dates <= test_timestamp) else []
                    after = dates[dates >= test_timestamp][:5] if any(dates >= test_timestamp) else []

                    if len(before) > 0:
                        print(f"  Closest dates before: {[d.date() for d in before]}")
                    if len(after) > 0:
                        print(f"  Closest dates after: {[d.date() for d in after]}")
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Q
from markets.models import Symbol, OHLCVData, PrecomputedMetrics
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Compute PrecomputedMetrics for all dates using concatenated historical data'

    def add_arguments(self, parser):
        parser.add_argument('--frequency', type=str, default='1D',
                          help='Frequency to process (default: 1D)')
        parser.add_argument('--symbol', type=str,
                          help='Specific symbol to process')
        parser.add_argument('--all-symbols', action='store_true',
                          help='Process all symbols')
        parser.add_argument('--start-date', type=str,
                          help='Start date for metrics calculation (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=str,
                          help='End date for metrics calculation (YYYY-MM-DD)')
        parser.add_argument('--batch-size', type=int, default=1000,
                          help='Batch size for bulk create (default: 1000)')
        parser.add_argument('--delete-existing', action='store_true',
                          help='Delete existing metrics before processing')

    def handle(self, *args, **options):
        frequency = options.get('frequency', '1D')
        specific_symbol = options.get('symbol')
        all_symbols = options.get('all_symbols')
        start_date_str = options.get('start_date')
        end_date_str = options.get('end_date')
        batch_size = options.get('batch_size', 1000)
        delete_existing = options.get('delete_existing', False)

        # Parse dates if provided
        start_date = None
        if start_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()

        end_date = None
        if end_date_str:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        # Determine which symbols to process
        if specific_symbol:
            symbols = Symbol.objects.filter(ticker=specific_symbol, is_active=True)
            if not symbols.exists():
                self.stdout.write(self.style.ERROR(f"Symbol {specific_symbol} not found"))
                return
        elif all_symbols:
            symbols = Symbol.objects.filter(is_active=True).order_by('ticker')
        else:
            self.stdout.write(self.style.WARNING(
                "Please specify --symbol or --all-symbols"
            ))
            return

        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"COMPUTING METRICS FOR {frequency} DATA")
        self.stdout.write(f"{'='*60}")
        self.stdout.write(f"Symbols to process: {symbols.count()}")
        if start_date:
            self.stdout.write(f"Start date: {start_date}")
        if end_date:
            self.stdout.write(f"End date: {end_date}")
        self.stdout.write(f"Batch size: {batch_size}")
        if delete_existing:
            self.stdout.write(self.style.WARNING("Will delete existing metrics!"))
        self.stdout.write(f"{'='*60}\n")

        total_metrics_created = 0
        symbols_processed = 0
        symbols_skipped = 0

        for symbol in symbols:
            self.stdout.write(f"\nProcessing {symbol.ticker}...")

            try:
                # Get ALL OHLCV data for this symbol across all years
                ohlcv_data = OHLCVData.objects.filter(
                    symbol=symbol,
                    frequency=frequency
                ).order_by('year')

                if not ohlcv_data.exists():
                    self.stdout.write(self.style.WARNING(f"  No OHLCV data found for {symbol.ticker}"))
                    symbols_skipped += 1
                    continue

                # Concatenate data from all years
                self.stdout.write(f"  Loading data from {ohlcv_data.count()} years...")
                dfs = []

                for data in ohlcv_data:
                    df = data.get_data_as_dataframe()
                    if df is not None and not df.empty:
                        dfs.append(df)

                if not dfs:
                    self.stdout.write(self.style.WARNING(f"  No valid data for {symbol.ticker}"))
                    symbols_skipped += 1
                    continue

                # Concatenate all data and sort chronologically
                combined_df = pd.concat(dfs)
                combined_df.sort_index(inplace=True)

                # Remove any duplicate dates (shouldn't happen, but just in case)
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

                if symbols_processed == 0:  # Only for first symbol
                    self.diagnose_data(symbol, combined_df)

                total_days = len(combined_df)
                self.stdout.write(f"  Total historical data: {total_days} days")
                self.stdout.write(f"  Date range: {combined_df.index[0].date()} to {combined_df.index[-1].date()}")

                # Filter by date range if specified - handle timezone-aware indices
                if start_date:
                    # Convert start_date to timezone-aware Timestamp at start of day
                    start_timestamp = pd.Timestamp(start_date).tz_localize('UTC')
                    self.stdout.write(f"  Filtering from: {start_timestamp}")
                    before_count = len(combined_df)
                    combined_df = combined_df[combined_df.index >= start_timestamp]
                    after_count = len(combined_df)
                    self.stdout.write(f"  Removed {before_count - after_count} days before {start_date}")

                if end_date:
                    # Convert end_date to timezone-aware Timestamp at end of day
                    end_timestamp = pd.Timestamp(end_date).tz_localize('UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    self.stdout.write(f"  Filtering to: {end_timestamp}")
                    before_count = len(combined_df)
                    combined_df = combined_df[combined_df.index <= end_timestamp]
                    after_count = len(combined_df)
                    self.stdout.write(f"  Removed {before_count - after_count} days after {end_date}")

                if combined_df.empty:
                    self.stdout.write(self.style.WARNING(f"  No data in specified date range"))
                    symbols_skipped += 1
                    continue

                # Delete existing metrics if requested
                if delete_existing:
                    deleted = PrecomputedMetrics.objects.filter(
                        symbol=symbol,
                        frequency=frequency
                    ).delete()
                    self.stdout.write(f"  Deleted {deleted[0]} existing metrics")

                # Compute metrics for all dates
                metrics_created = self.compute_metrics_for_symbol(
                    symbol,
                    frequency,
                    combined_df,
                    batch_size
                )

                total_metrics_created += metrics_created
                symbols_processed += 1

                self.stdout.write(self.style.SUCCESS(
                    f"  ✓ Created {metrics_created} metrics records for {symbol.ticker}"
                ))

            except Exception as e:
                self.stdout.write(self.style.ERROR(
                    f"  ✗ Error processing {symbol.ticker}: {str(e)}"
                ))
                logger.error(f"Error processing {symbol.ticker}: {str(e)}", exc_info=True)

        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"COMPLETED!\n"
            f"{'='*60}\n"
            f"Symbols processed: {symbols_processed}\n"
            f"Symbols skipped: {symbols_skipped}\n"
            f"Total metrics created: {total_metrics_created}\n"
            f"{'='*60}"
        ))

    def compute_metrics_for_symbol(self, symbol, frequency, df, batch_size):
        """Compute metrics for all dates in the dataframe using complete history"""

        # Define periods in days
        periods = {
            'change_1d': 1,
            'change_1w': 7,
            'change_2w': 14,
            'change_1m': 30,
            'change_3m': 90,
            'change_6m': 180,
            'change_1y': 365,
        }

        metrics_batch = []
        total_created = 0
        total_dates = len(df)

        self.stdout.write(f"  Computing metrics for {total_dates} dates...")

        for idx, (current_date, row) in enumerate(df.iterrows()):
            current_price = float(row['close'])

            # Calculate changes for this date using ALL historical data
            changes = {}

            for field, days in periods.items():
                try:
                    target_date = current_date - pd.Timedelta(days=days)

                    # Find the closest date that's not after target_date
                    # Using all available historical data, not just current year
                    available_dates = df.index[df.index <= target_date]

                    if len(available_dates) > 0:
                        closest_date = available_dates[-1]
                        past_price = float(df.loc[closest_date, 'close'])

                        if past_price and past_price > 0:
                            pct_change = ((current_price - past_price) / past_price) * 100
                            changes[field] = round(pct_change, 2)
                        else:
                            changes[field] = None
                    else:
                        # Not enough historical data
                        changes[field] = None

                except Exception as e:
                    logger.warning(f"Error calculating {field} for {symbol.ticker} on {current_date}: {str(e)}")
                    changes[field] = None

            # Create metrics object
            metrics_batch.append(
                PrecomputedMetrics(
                    symbol=symbol,
                    frequency=frequency,
                    as_of_date=current_date.date(),  # Convert to date (timezone-naive)
                    current_price=current_price,
                    change_1d=changes.get('change_1d'),
                    change_1w=changes.get('change_1w'),
                    change_2w=changes.get('change_2w'),
                    change_1m=changes.get('change_1m'),
                    change_3m=changes.get('change_3m'),
                    change_6m=changes.get('change_6m'),
                    change_1y=changes.get('change_1y'),
                )
            )

            # Bulk create in batches
            if len(metrics_batch) >= batch_size or idx == total_dates - 1:
                PrecomputedMetrics.objects.bulk_create(
                    metrics_batch,
                    ignore_conflicts=True  # Skip if record already exists
                )
                total_created += len(metrics_batch)
                metrics_batch = []

                if idx % 1000 == 0 and idx > 0:
                    self.stdout.write(f"    Progress: {idx + 1}/{total_dates} dates processed")

        return total_created

    def diagnose_data(self, symbol, df):
        """Diagnose data continuity for a symbol"""
        self.stdout.write("\n  " + "="*50)
        self.stdout.write("  DATA DIAGNOSTICS")
        self.stdout.write("  " + "="*50)

        # Show date range
        self.stdout.write(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        self.stdout.write(f"  Total days: {len(df)}")

        # Check for gaps
        date_diff = df.index.to_series().diff().dt.days
        gaps = date_diff[date_diff > 1]
        if len(gaps) > 0:
            self.stdout.write(self.style.WARNING(f"  Found {len(gaps)} gaps:"))
            for idx in gaps.index[:5]:  # Show first 5 gaps
                prev_date = df.index[df.index.get_loc(idx) - 1]
                self.stdout.write(f"    {prev_date.date()} -> {idx.date()}: {int(gaps[idx])} days")

        # Check specific date
        test_date = pd.Timestamp('2026-01-30').tz_localize('UTC')
        if test_date in df.index:
            self.stdout.write(self.style.SUCCESS(f"  Test date 2026-01-30 found in data"))

            # Check historical data availability
            price = df.loc[test_date, 'close']
            self.stdout.write(f"  Price on 2026-01-30: {price}")

            # Check 1M ago
            target_1m = test_date - pd.Timedelta(days=30)
            available_1m = df.index[df.index <= target_1m]
            if len(available_1m) > 0:
                closest_1m = available_1m[-1]
                price_1m = df.loc[closest_1m, 'close']
                days_diff = (test_date - closest_1m).days
                self.stdout.write(f"  1M ago: {closest_1m.date()} ({days_diff} days earlier), price: {price_1m}")
            else:
                self.stdout.write(self.style.WARNING(f"  No data found for 1M ago"))

            # Check 6M ago
            target_6m = test_date - pd.Timedelta(days=180)
            available_6m = df.index[df.index <= target_6m]
            if len(available_6m) > 0:
                closest_6m = available_6m[-1]
                price_6m = df.loc[closest_6m, 'close']
                days_diff = (test_date - closest_6m).days
                self.stdout.write(f"  6M ago: {closest_6m.date()} ({days_diff} days earlier), price: {price_6m}")
            else:
                self.stdout.write(self.style.WARNING(f"  No data found for 6M ago"))

            # Check 1Y ago
            target_1y = test_date - pd.Timedelta(days=365)
            available_1y = df.index[df.index <= target_1y]
            if len(available_1y) > 0:
                closest_1y = available_1y[-1]
                price_1y = df.loc[closest_1y, 'close']
                days_diff = (test_date - closest_1y).days
                self.stdout.write(f"  1Y ago: {closest_1y.date()} ({days_diff} days earlier), price: {price_1y}")
            else:
                self.stdout.write(self.style.WARNING(f"  No data found for 1Y ago"))
        else:
            self.stdout.write(self.style.WARNING(f"  Test date 2026-01-30 NOT found in data"))

        self.stdout.write("  " + "="*50)

