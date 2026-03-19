import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone
from markets.models import Symbol, OHLCVData, PrecomputedMetrics
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    '''
        python manage.py load_ohlcv_data --frequency 1D --year 2026 --data-dir ../ibkr/data/
    '''
    help = 'Load OHLCV data from CSV files into the database'

    def add_arguments(self, parser):
        parser.add_argument('--frequency', type=str, help='Frequency to load (1min, 5min, 15min, 1H, 4H, 1D)')
        parser.add_argument('--year', type=int, help='Year to load')
        parser.add_argument('--symbol', type=str, help='Specific symbol to load')
        parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
        parser.add_argument('--data-dir', type=str, help='Override data directory')

    def handle(self, *args, **options):
        frequency = options.get('frequency')
        year = options.get('year')
        specific_symbol = options.get('symbol')
        batch_size = options.get('batch_size')
        data_dir_override = options.get('data_dir')

        # Get data directory
        if data_dir_override:
            data_dir = Path(data_dir_override)
        else:
            data_dir = Path(settings.BASE_DIR) / 'data'

        self.stdout.write(f"Using data directory: {data_dir}")

        if not data_dir.exists():
            self.stdout.write(self.style.ERROR(f"Data directory not found: {data_dir}"))
            return

        if frequency:
            frequencies = [frequency]
        else:
            frequencies = ['1min', '5min', '15min', '1H', '4H', '1D']

        if year:
            years = [year]
        else:
            # Get all available years from directory structure
            years = []
            for freq in frequencies:
                freq_path = data_dir / freq
                if freq_path.exists():
                    years.extend([int(d.name) for d in freq_path.iterdir() if d.is_dir()])
            years = sorted(set(years))

        if not years:
            self.stdout.write(self.style.WARNING("No years found to process"))
            return

        self.stdout.write(f"Processing frequencies: {frequencies}")
        self.stdout.write(f"Processing years: {years}")

        total_loaded = 0
        total_errors = 0

        for freq in frequencies:
            for yr in years:
                self.stdout.write(f"\nProcessing {freq} - {yr}...")

                freq_year_path = data_dir / freq / str(yr)
                if not freq_year_path.exists():
                    self.stdout.write(f"  Directory not found: {freq_year_path}")
                    continue

                # Get all CSV files
                csv_files = list(freq_year_path.glob("*.csv"))
                csv_files.extend(freq_year_path.glob("*.CSV"))  # Also catch uppercase extensions

                if specific_symbol:
                    csv_files = [f for f in csv_files if f.stem.split('_')[0].upper() == specific_symbol.upper()]

                if not csv_files:
                    self.stdout.write(f"  No CSV files found in {freq_year_path}")
                    continue

                self.stdout.write(f"  Found {len(csv_files)} files to process")

                for i, csv_file in enumerate(csv_files, 1):
                    try:
                        # Extract symbol name from filename (assumes format: SYMBOL_YYYY.csv)
                        symbol_name = csv_file.stem.split('_')[0].upper()

                        self.stdout.write(f"    Processing {i}/{len(csv_files)}: {symbol_name}")

                        # Get or create symbol
                        symbol, created = Symbol.objects.get_or_create(
                            ticker=symbol_name,
                            defaults={
                                'name': symbol_name,
                                'is_active': True
                            }
                        )

                        if created:
                            self.stdout.write(f"      Created new symbol: {symbol_name}")

                        # Load and process CSV
                        try:
                            df = pd.read_csv(csv_file)
                        except Exception as e:
                            self.stdout.write(self.style.ERROR(
                                f"      Error reading CSV {csv_file.name}: {str(e)}"
                            ))
                            total_errors += 1
                            continue

                        if df.empty:
                            self.stdout.write(self.style.WARNING(
                                f"      Skipping {csv_file.name}: empty file"
                            ))
                            continue

                        # Standardize column names (assuming standard OHLCV format)
                        column_mapping = {
                            'date': 'dates',
                            'Date': 'dates',
                            'DATE': 'dates',
                            'timestamp': 'dates',
                            'Timestamp': 'dates',
                            'TIMESTAMP': 'dates',
                            'datetime': 'dates',
                            'Datetime': 'dates',
                            'DATETIME': 'dates',
                            'open': 'open',
                            'Open': 'open',
                            'OPEN': 'open',
                            'high': 'high',
                            'High': 'high',
                            'HIGH': 'high',
                            'low': 'low',
                            'Low': 'low',
                            'LOW': 'low',
                            'close': 'close',
                            'Close': 'close',
                            'CLOSE': 'close',
                            'volume': 'volume',
                            'Volume': 'volume',
                            'VOLUME': 'volume',
                        }

                        # Rename columns that exist
                        rename_dict = {col: column_mapping[col] for col in df.columns if col in column_mapping}
                        if rename_dict:
                            df.rename(columns=rename_dict, inplace=True)

                        # Check for required columns
                        required_cols = ['dates', 'open', 'high', 'low', 'close', 'volume']
                        missing_cols = [col for col in required_cols if col not in df.columns]

                        if missing_cols:
                            self.stdout.write(self.style.WARNING(
                                f"      Skipping {csv_file.name}: missing columns {missing_cols}"
                            ))
                            self.stdout.write(f"      Available columns: {list(df.columns)}")
                            total_errors += 1
                            continue

                        # Convert dates
                        try:
                            df['dates'] = pd.to_datetime(df['dates'])
                        except Exception as e:
                            self.stdout.write(self.style.ERROR(
                                f"      Error converting dates in {csv_file.name}: {str(e)}"
                            ))
                            total_errors += 1
                            continue

                        # Remove any rows with NaN values
                        initial_rows = len(df)
                        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
                        if len(df) < initial_rows:
                            self.stdout.write(f"      Removed {initial_rows - len(df)} rows with NaN values")

                        if df.empty:
                            self.stdout.write(self.style.WARNING(
                                f"      Skipping {csv_file.name}: no valid data after cleaning"
                            ))
                            continue

                        # Sort by date
                        df.sort_values('dates', inplace=True)

                        # Prepare data for JSON storage
                        data_dict = {
                            'dates': df['dates'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                            'open': [float(x) for x in df['open'].tolist()],
                            'high': [float(x) for x in df['high'].tolist()],
                            'low': [float(x) for x in df['low'].tolist()],
                            'close': [float(x) for x in df['close'].tolist()],
                            'volume': [float(x) for x in df['volume'].tolist()],
                        }

                        # Create or update OHLCV data
                        with transaction.atomic():
                            ohlcv_data, created = OHLCVData.objects.update_or_create(
                                symbol=symbol,
                                frequency=freq,
                                year=yr,
                                defaults={
                                    'data': data_dict,
                                    'first_date': df['dates'].min().date(),
                                    'last_date': df['dates'].max().date(),
                                    'total_records': len(df),
                                }
                            )

                            if created:
                                self.stdout.write(f"      Created new OHLCV record for {symbol_name}")
                            else:
                                self.stdout.write(f"      Updated OHLCV record for {symbol_name}")

                            # Precompute metrics for the latest date
                            self.precompute_metrics(symbol, freq, df)

                            total_loaded += 1

                            if total_loaded % batch_size == 0:
                                self.stdout.write(self.style.SUCCESS(
                                    f"    Loaded {total_loaded} files so far..."
                                ))

                    except Exception as e:
                        self.stdout.write(self.style.ERROR(
                            f"    Error processing {csv_file.name}: {str(e)}"
                        ))
                        logger.error(f"Error loading {csv_file}: {str(e)}", exc_info=True)
                        total_errors += 1

        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*50}\n"
            f"Loading complete!\n"
            f"Successfully loaded: {total_loaded} files\n"
            f"Errors encountered: {total_errors} files\n"
            f"{'='*50}"
        ))

    def precompute_metrics(self, symbol, frequency, df):
        """Precompute percentage changes for the latest date"""
        if len(df) == 0:
            return

        try:
            # Get latest date and price
            latest_idx = df['dates'].idxmax()
            latest_date = df.loc[latest_idx, 'dates']
            latest_price = float(df.loc[latest_idx, 'close'])

            # Convert dates to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['dates']):
                df['dates'] = pd.to_datetime(df['dates'])

            # Create a Series with dates as index for easier lookup
            df_with_index = df.set_index('dates').sort_index()

            # Calculate changes for different periods
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

            for field, days in periods.items():
                try:
                    target_date = latest_date - pd.Timedelta(days=days)

                    # Find the closest date
                    closest_idx = df_with_index.index.get_indexer([target_date], method='nearest')[0]
                    if closest_idx >= 0:
                        past_price = float(df_with_index.iloc[closest_idx]['close'])

                        if past_price and past_price > 0:
                            pct_change = ((latest_price - past_price) / past_price) * 100
                            changes[field] = round(pct_change, 2)
                        else:
                            changes[field] = None
                    else:
                        changes[field] = None
                except Exception as e:
                    logger.warning(f"Error calculating {field} for {symbol.ticker}: {str(e)}")
                    changes[field] = None

            # Create or update metrics
            metrics, created = PrecomputedMetrics.objects.update_or_create(
                symbol=symbol,
                frequency=frequency,
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

            if created:
                logger.info(f"Created metrics for {symbol.ticker} as of {latest_date.date()}")

        except Exception as e:
            logger.error(f"Error precomputing metrics for {symbol.ticker}: {str(e)}", exc_info=True)

