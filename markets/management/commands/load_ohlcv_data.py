import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from markets.models import Symbol, OHLCVData, PrecomputedMetrics
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    '''
    Usage:
        python manage.py load_ohlcv_data --frequency 1D --year 2026 --data-dir ../ibkr/data/
        python manage.py load_ohlcv_data --frequency 1D --year 2026 --symbol FMCC --data-dir ../ibkr/data/
        python manage.py load_ohlcv_data --frequency 1D --year 2026 --delete-existing --data-dir ../ibkr/data/
    '''
    help = 'Load OHLCV data from CSV files into the database'

    def add_arguments(self, parser):
        parser.add_argument('--frequency', type=str, help='Frequency to load (1min, 5min, 15min, 1H, 4H, 1D)')
        parser.add_argument('--year', type=int, help='Year to load')
        parser.add_argument('--symbol', type=str, help='Specific symbol to load')
        parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
        parser.add_argument('--data-dir', type=str, help='Override data directory')
        parser.add_argument('--delete-existing', action='store_true', help='Delete existing records before loading')

    def handle(self, *args, **options):
        frequency = options.get('frequency')
        year = options.get('year')
        specific_symbol = options.get('symbol')
        batch_size = options.get('batch_size')
        data_dir_override = options.get('data_dir')
        delete_existing = options.get('delete_existing')

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
                    years.extend([int(d.name) for d in freq_path.iterdir() if d.is_dir() and d.name.isdigit()])
            years = sorted(set(years))

        if not years:
            self.stdout.write(self.style.WARNING("No years found to process"))
            return

        self.stdout.write(f"Processing frequencies: {frequencies}")
        self.stdout.write(f"Processing years: {years}")

        # Delete existing records if requested
        if delete_existing:
            self.delete_existing_records(frequencies, years, specific_symbol)
            self.stdout.write(self.style.WARNING("Existing records deleted"))

        total_loaded = 0
        total_errors = 0
        total_skipped = 0

        for freq in frequencies:
            for yr in years:
                self.stdout.write(f"\n{'='*60}")
                self.stdout.write(f"Processing {freq} - {yr}...")
                self.stdout.write('='*60)

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

                        self.stdout.write(f"\n  [{i}/{len(csv_files)}] Processing: {symbol_name}")

                        # Get or create symbol
                        symbol, created = Symbol.objects.get_or_create(
                            ticker=symbol_name,
                            defaults={
                                'name': symbol_name,
                                'is_active': True
                            }
                        )

                        if created:
                            self.stdout.write(f"    Created new symbol: {symbol_name}")

                        # Load and process CSV with proper date handling
                        result = self.process_csv_file(csv_file, symbol, freq, yr)

                        if result['success']:
                            total_loaded += 1
                            self.stdout.write(self.style.SUCCESS(
                                f"    ✓ Loaded {result['records']} records for {symbol_name}"
                            ))
                            if result['warnings']:
                                for warning in result['warnings']:
                                    self.stdout.write(self.style.WARNING(f"      ⚠ {warning}"))
                        else:
                            total_errors += 1
                            self.stdout.write(self.style.ERROR(
                                f"    ✗ Failed to load {symbol_name}: {result['error']}"
                            ))

                        if total_loaded % batch_size == 0 and total_loaded > 0:
                            self.stdout.write(self.style.SUCCESS(
                                f"\n  Progress: Loaded {total_loaded} files so far..."
                            ))

                    except Exception as e:
                        self.stdout.write(self.style.ERROR(
                            f"    ✗ Unexpected error processing {csv_file.name}: {str(e)}"
                        ))
                        logger.error(f"Error loading {csv_file}: {str(e)}", exc_info=True)
                        total_errors += 1

        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"LOADING COMPLETE!\n"
            f"{'='*60}\n"
            f"Successfully loaded: {total_loaded} files\n"
            f"Errors encountered: {total_errors} files\n"
            f"Skipped: {total_skipped} files\n"
            f"{'='*60}"
        ))

    def delete_existing_records(self, frequencies, years, specific_symbol):
        """Delete existing records for the given parameters"""
        queryset = OHLCVData.objects.filter(
            frequency__in=frequencies,
            year__in=years
        )

        if specific_symbol:
            queryset = queryset.filter(symbol__ticker=specific_symbol)

        count = queryset.count()
        queryset.delete()

        # Also delete related metrics
        metrics_queryset = PrecomputedMetrics.objects.filter(
            frequency__in=frequencies
        )
        if specific_symbol:
            metrics_queryset = metrics_queryset.filter(symbol__ticker=specific_symbol)
        metrics_queryset.delete()

        self.stdout.write(f"  Deleted {count} existing OHLCV records")

    def process_csv_file(self, csv_file, symbol, frequency, year):
        """Process a single CSV file and load into database"""
        result = {
            'success': False,
            'records': 0,
            'warnings': [],
            'error': None
        }

        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                result['error'] = "Could not read CSV with any encoding"
                return result

            if df.empty:
                result['error'] = "Empty file"
                return result

            # Standardize column names
            df = self.standardize_columns(df)

            # Check for required columns
            required_cols = ['dates', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                result['error'] = f"Missing columns: {missing_cols}"
                return result

            # Convert dates with robust parsing
            df = self.parse_dates_robust(df)

            if df.empty:
                result['error'] = "No valid dates after parsing"
                return result

            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with NaN values in critical columns
            initial_rows = len(df)
            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

            if len(df) < initial_rows:
                result['warnings'].append(f"Removed {initial_rows - len(df)} rows with NaN values")

            if df.empty:
                result['error'] = "No valid data after cleaning"
                return result

            # Sort by date
            df.sort_values('dates', inplace=True)

            # Remove duplicates
            initial_rows = len(df)
            df.drop_duplicates(subset=['dates'], keep='first', inplace=True)

            if len(df) < initial_rows:
                result['warnings'].append(f"Removed {initial_rows - len(df)} duplicate dates")

            # Prepare data for JSON storage with validated dates
            data_dict = {
                'dates': [d.strftime('%Y-%m-%dT%H:%M:%S') for d in df['dates'].tolist()],
                'open': [float(x) for x in df['open'].tolist()],
                'high': [float(x) for x in df['high'].tolist()],
                'low': [float(x) for x in df['low'].tolist()],
                'close': [float(x) for x in df['close'].tolist()],
                'volume': [float(x) for x in df['volume'].tolist()],
            }

            # Validate dates before saving
            valid_dates = []
            for date_str in data_dict['dates'][:10]:  # Check first 10
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', ''))
                    valid_dates.append(dt)
                except ValueError as e:
                    result['error'] = f"Invalid date format: {date_str} - {str(e)}"
                    return result

            if not valid_dates:
                result['error'] = "No valid dates found"
                return result

            # Get min and max dates
            first_date = min(valid_dates).date()
            last_date = max(valid_dates).date()

            # Verify dates are within expected year
            if first_date.year != year and last_date.year != year:
                result['warnings'].append(
                    f"Date range {first_date} to {last_date} spans outside year {year}"
                )

            # Create or update OHLCV data
            with transaction.atomic():
                ohlcv_data, created = OHLCVData.objects.update_or_create(
                    symbol=symbol,
                    frequency=frequency,
                    year=year,
                    defaults={
                        'data': data_dict,
                        'first_date': first_date,
                        'last_date': last_date,
                        'total_records': len(df),
                    }
                )

                # Precompute metrics for the latest date
                self.precompute_metrics(symbol, frequency, df)

                result['success'] = True
                result['records'] = len(df)
                result['created'] = created

                # Add date range to result
                result['date_range'] = f"{first_date} to {last_date}"

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing {csv_file}: {str(e)}", exc_info=True)

        return result

    def standardize_columns(self, df):
        """Standardize column names to expected format"""
        column_mapping = {
            # Date variations
            'date': 'dates',
            'Date': 'dates',
            'DATE': 'dates',
            'timestamp': 'dates',
            'Timestamp': 'dates',
            'TIMESTAMP': 'dates',
            'datetime': 'dates',
            'Datetime': 'dates',
            'DATETIME': 'dates',
            'time': 'dates',
            'Time': 'dates',
            'TIME': 'dates',

            # Open variations
            'open': 'open',
            'Open': 'open',
            'OPEN': 'open',
            'opening': 'open',
            'Opening': 'open',

            # High variations
            'high': 'high',
            'High': 'high',
            'HIGH': 'high',
            'max': 'high',
            'Max': 'high',

            # Low variations
            'low': 'low',
            'Low': 'low',
            'LOW': 'low',
            'min': 'low',
            'Min': 'low',

            # Close variations
            'close': 'close',
            'Close': 'close',
            'CLOSE': 'close',
            'closing': 'close',
            'Closing': 'close',
            'price': 'close',
            'Price': 'close',

            # Volume variations
            'volume': 'volume',
            'Volume': 'volume',
            'VOLUME': 'volume',
            'vol': 'volume',
            'Vol': 'volume',
            'VOL': 'volume',
            'trading volume': 'volume',
            'Trading Volume': 'volume',
        }

        # Rename columns that exist
        rename_dict = {col: column_mapping[col] for col in df.columns if col in column_mapping}
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)

        return df

    def parse_dates_robust(self, df):
        """Robust date parsing with multiple format attempts"""

        def try_parse_date(date_value):
            """Try multiple date parsing strategies"""
            if pd.isna(date_value):
                return None

            # If it's already a datetime, return it
            if isinstance(date_value, (pd.Timestamp, datetime)):
                return date_value

            # Convert to string for parsing
            date_str = str(date_value).strip()

            # Try different formats
            formats = [
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%d-%m-%Y',
                '%d/%m/%Y',
                '%m-%d-%Y',
                '%m/%d/%Y',
                '%Y%m%d',
                '%d-%b-%Y',
                '%d/%b/%Y',
                '%b %d, %Y',
                '%B %d, %Y',
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S',
                '%d-%m-%Y %H:%M:%S',
                '%d/%m/%Y %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S.%f',
            ]

            # Try each format
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            # Try pandas parsing as last resort
            try:
                return pd.to_datetime(date_str, errors='coerce')
            except:
                return None

        # Apply parsing to dates column
        df['dates'] = df['dates'].apply(try_parse_date)

        # Remove rows with invalid dates
        df = df[df['dates'].notna()]

        return df

    def precompute_metrics(self, symbol, frequency, df):
        """Precompute percentage changes for the latest date"""
        if len(df) == 0:
            return

        try:
            # Get latest date and price
            latest_idx = df['dates'].idxmax()
            latest_date = df.loc[latest_idx, 'dates']
            latest_price = float(df.loc[latest_idx, 'close'])

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

                    if closest_idx >= 0 and closest_idx < len(df_with_index):
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
            PrecomputedMetrics.objects.update_or_create(
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

        except Exception as e:
            logger.error(f"Error precomputing metrics for {symbol.ticker}: {str(e)}", exc_info=True)

