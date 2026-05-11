import pandas as pd
import numpy as np
import talib
from django.core.management.base import BaseCommand
from django.db import transaction
from markets.models import Symbol, OHLCVData, PrecomputedMetrics, TechnicalIndicators
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Compute technical indicators for all symbols'

    def add_arguments(self, parser):
        parser.add_argument('--frequency', type=str, default='1D',
                          help='Frequency to process (1D, 1H, etc.) (default: 1D)')
        parser.add_argument('--symbol', type=str, help='Specific symbol to process')
        parser.add_argument('--all-symbols', action='store_true', help='Process all symbols')
        parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
        parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
        parser.add_argument('--delete-existing', action='store_true',
                          help='Delete existing indicators before processing')

    def handle(self, *args, **options):
        frequency = options.get('frequency', '1D')
        symbol_filter = options.get('symbol')
        all_symbols = options.get('all_symbols')
        start_date_str = options.get('start_date')
        end_date_str = options.get('end_date')
        batch_size = options.get('batch_size', 100)
        delete_existing = options.get('delete_existing', False)

        # Parse dates if provided
        start_date = None
        if start_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()

        end_date = None
        if end_date_str:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        if symbol_filter:
            symbols = Symbol.objects.filter(ticker=symbol_filter)
        elif all_symbols:
            symbols = Symbol.objects.filter(is_active=True)
        else:
            self.stdout.write("Please specify --symbol or --all-symbols")
            return

        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"COMPUTING TECHNICAL INDICATORS FOR {frequency}")
        self.stdout.write(f"{'='*60}")
        self.stdout.write(f"Symbols to process: {symbols.count()}")
        self.stdout.write(f"Frequency: {frequency}")
        if start_date:
            self.stdout.write(f"Start date: {start_date}")
        if end_date:
            self.stdout.write(f"End date: {end_date}")
        if delete_existing:
            self.stdout.write(self.style.WARNING("Will delete existing indicators!"))
        self.stdout.write(f"{'='*60}\n")

        total_indicators_created = 0
        symbols_processed = 0
        symbols_skipped = 0

        for symbol in symbols:
            self.stdout.write(f"\nProcessing {symbol.ticker}...")
            try:
                indicators_created = self.compute_indicators_for_symbol(
                    symbol, frequency, start_date, end_date, batch_size, delete_existing
                )
                if indicators_created > 0:
                    total_indicators_created += indicators_created
                    symbols_processed += 1
                    self.stdout.write(self.style.SUCCESS(
                        f"  ✓ Created {indicators_created} technical indicator records for {symbol.ticker}"
                    ))
                else:
                    symbols_skipped += 1
                    self.stdout.write(self.style.WARNING(f"  No indicators created for {symbol.ticker}"))
            except Exception as e:
                symbols_skipped += 1
                self.stdout.write(self.style.ERROR(f"  ✗ Error: {str(e)}"))
                logger.error(f"Error processing {symbol.ticker}: {str(e)}", exc_info=True)

        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"COMPLETED!\n"
            f"{'='*60}\n"
            f"Symbols processed: {symbols_processed}\n"
            f"Symbols skipped: {symbols_skipped}\n"
            f"Total technical indicator records created: {total_indicators_created}\n"
            f"{'='*60}"
        ))

    def compute_indicators_for_symbol(self, symbol, frequency, start_date, end_date, batch_size, delete_existing):
        """Compute all technical indicators for a symbol"""

        self.stdout.write(f"  Checking symbol: {symbol.ticker}")

        # First check if PrecomputedMetrics exists
        metrics_qs = PrecomputedMetrics.objects.filter(
            symbol=symbol,
            frequency=frequency
        ).order_by('as_of_date')

        if start_date:
            metrics_qs = metrics_qs.filter(as_of_date__gte=start_date)
        if end_date:
            metrics_qs = metrics_qs.filter(as_of_date__lte=end_date)

        metrics_count = metrics_qs.count()
        self.stdout.write(f"  PrecomputedMetrics records found: {metrics_count}")

        if metrics_count == 0:
            self.stdout.write(f"  No PrecomputedMetrics found for {symbol.ticker} with frequency {frequency}")
            return 0

        # Check for OHLCV data
        ohlcv_data = OHLCVData.objects.filter(
            symbol=symbol,
            frequency=frequency
        ).order_by('year')

        ohlcv_count = ohlcv_data.count()
        self.stdout.write(f"  OHLCV data records found: {ohlcv_count}")

        if ohlcv_count == 0:
            self.stdout.write(self.style.WARNING(
                f"  No OHLCV data found for {symbol.ticker}. "
                f"Please load OHLCV data first with: "
                f"python manage.py load_ohlcv_data --frequency {frequency} --symbol {symbol.ticker}"
            ))
            return 0

        # Delete existing indicators if requested
        if delete_existing:
            deleted = TechnicalIndicators.objects.filter(
                symbol=symbol,
                frequency=frequency
            ).delete()
            if deleted[0] > 0:
                self.stdout.write(f"  Deleted {deleted[0]} existing technical indicator records")

        # Concatenate all years
        dfs = []
        has_volume = True

        for data in ohlcv_data:
            df = data.get_data_as_dataframe()
            if df is not None and not df.empty:
                # Check if volume data exists and is non-zero
                if 'volume' in df.columns:
                    if df['volume'].sum() == 0:
                        # This is a special case (like VIX) - keep all data
                        if has_volume and not dfs:
                            self.stdout.write(f"  Note: {symbol.ticker} has zero volume data (expected for indices)")
                            has_volume = False
                        dfs.append(df)
                    else:
                        # Filter out zero-volume days only if there's some volume elsewhere
                        original_len = len(df)
                        df = df[df['volume'] > 0].copy()
                        if len(df) < original_len and len(df) > 0:
                            self.stdout.write(f"    Filtered out {original_len - len(df)} zero-volume days")
                        if not df.empty:
                            dfs.append(df)
                else:
                    dfs.append(df)

        if not dfs:
            self.stdout.write(f"  No valid OHLCV data after filtering for {symbol.ticker}")
            return 0

        combined_df = pd.concat(dfs)
        combined_df.sort_index(inplace=True)

        # Ensure we have all required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in combined_df.columns]
        if missing_cols:
            self.stdout.write(self.style.ERROR(f"  Missing columns: {missing_cols}"))
            return 0

        self.stdout.write(f"  Total historical OHLCV data: {len(combined_df)} days")
        self.stdout.write(f"  Date range: {combined_df.index[0].date()} to {combined_df.index[-1].date()}")

        # Check if we have data for the requested date range
        if start_date:
            start_timestamp = pd.Timestamp(start_date).tz_localize('UTC')
            if start_timestamp > combined_df.index[-1]:
                self.stdout.write(self.style.WARNING(
                    f"  Requested start date {start_date} is after last available data {combined_df.index[-1].date()}"
                ))
        if end_date:
            end_timestamp = pd.Timestamp(end_date).tz_localize('UTC')
            if end_timestamp < combined_df.index[0]:
                self.stdout.write(self.style.WARNING(
                    f"  Requested end date {end_date} is before first available data {combined_df.index[0].date()}"
                ))

        # Get benchmark data for beta (if frequency is 1D)
        benchmark_df = None
        if frequency == '1D':
            benchmark_df = self.get_benchmark_data()
            if benchmark_df is not None:
                self.stdout.write(f"  Loaded benchmark data for beta calculation")

        # Create TechnicalIndicators for each PrecomputedMetrics record
        indicators_batch = []
        indicators_created = 0
        skipped_count = 0
        error_count = 0

        for idx, metric in enumerate(metrics_qs):
            current_date = metric.as_of_date
            current_timestamp = pd.Timestamp(current_date).tz_localize('UTC')

            # Check if we have data for this date
            if current_timestamp not in combined_df.index:
                if skipped_count == 0:
                    before = combined_df.index[combined_df.index <= current_timestamp]
                    after = combined_df.index[combined_df.index >= current_timestamp]
                    self.stdout.write(f"    No OHLCV data for {current_date}")
                    if len(before) > 0:
                        self.stdout.write(f"      Closest date before: {before[-1].date()}")
                    if len(after) > 0:
                        self.stdout.write(f"      Closest date after: {after[0].date()}")
                skipped_count += 1
                continue

            # Get historical data up to this date
            historical_df = combined_df[combined_df.index <= current_timestamp]

            # Need at least 200 days of data for reliable indicators
            if len(historical_df) < 200:
                skipped_count += 1
                if skipped_count <= 5:  # Show first few skipped records
                    self.stdout.write(f"    Skipping {current_date}: insufficient data ({len(historical_df)} days)")
                continue

            # Calculate indicators
            try:
                indicators = self.calculate_indicators(historical_df, current_timestamp, benchmark_df, has_volume)

                # Create TechnicalIndicators object linked to PrecomputedMetrics
                indicators_batch.append(
                    TechnicalIndicators(
                        precomputed_metrics=metric,
                        symbol=symbol,
                        frequency=frequency,
                        as_of_date=current_date,
                        **indicators
                    )
                )

                if len(indicators_batch) >= batch_size:
                    TechnicalIndicators.objects.bulk_create(
                        indicators_batch, ignore_conflicts=True
                    )
                    indicators_created += len(indicators_batch)
                    self.stdout.write(f"    Processed {indicators_created} records...")
                    indicators_batch = []

            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Show first few errors
                    self.stdout.write(f"    Error calculating indicators for {current_date}: {str(e)}")
                continue

            # Show progress every 100 records
            if idx > 0 and idx % 100 == 0:
                self.stdout.write(f"    Progress: {idx}/{metrics_qs.count()} records processed")

        # Save remaining
        if indicators_batch:
            TechnicalIndicators.objects.bulk_create(indicators_batch, ignore_conflicts=True)
            indicators_created += len(indicators_batch)

        if skipped_count > 0:
            self.stdout.write(f"  Skipped {skipped_count} records (insufficient historical data)")
        if error_count > 0:
            self.stdout.write(f"  {error_count} records had calculation errors")

        return indicators_created

    def calculate_indicators(self, df, current_date, benchmark_df=None, has_volume=True):
        """Calculate all technical indicators for a specific date"""
        # Get price arrays
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        open_prices = df['open'].values.astype(float)

        # Only use volume if available and has_volume flag is True
        if has_volume and 'volume' in df.columns and df['volume'].sum() > 0:
            volume = df['volume'].values.astype(float)
        else:
            volume = None

        current_idx = len(df) - 1
        current_price = close[current_idx]

        indicators = {}

        # Moving Averages (only if we have enough data)
        indicators['sma_9'] = self.safe_float(talib.SMA(close, timeperiod=9)[current_idx]) if len(close) >= 9 else None
        indicators['sma_20'] = self.safe_float(talib.SMA(close, timeperiod=20)[current_idx]) if len(close) >= 20 else None
        indicators['sma_30'] = self.safe_float(talib.SMA(close, timeperiod=30)[current_idx]) if len(close) >= 30 else None
        indicators['sma_50'] = self.safe_float(talib.SMA(close, timeperiod=50)[current_idx]) if len(close) >= 50 else None
        indicators['sma_100'] = self.safe_float(talib.SMA(close, timeperiod=100)[current_idx]) if len(close) >= 100 else None
        indicators['sma_200'] = self.safe_float(talib.SMA(close, timeperiod=200)[current_idx]) if len(close) >= 200 else None

        indicators['ema_9'] = self.safe_float(talib.EMA(close, timeperiod=9)[current_idx]) if len(close) >= 9 else None
        indicators['ema_20'] = self.safe_float(talib.EMA(close, timeperiod=20)[current_idx]) if len(close) >= 20 else None
        indicators['ema_30'] = self.safe_float(talib.EMA(close, timeperiod=30)[current_idx]) if len(close) >= 30 else None
        indicators['ema_50'] = self.safe_float(talib.EMA(close, timeperiod=50)[current_idx]) if len(close) >= 50 else None
        indicators['ema_100'] = self.safe_float(talib.EMA(close, timeperiod=100)[current_idx]) if len(close) >= 100 else None
        indicators['ema_200'] = self.safe_float(talib.EMA(close, timeperiod=200)[current_idx]) if len(close) >= 200 else None

        # Oscillators
        indicators['rsi_14'] = self.safe_float(talib.RSI(close, timeperiod=14)[current_idx]) if len(close) >= 14 else None

        # Stochastic - Fast
        if len(high) >= 14:
            fastk, fastd = talib.STOCHF(high, low, close, fastk_period=14, fastd_period=3)
            indicators['stoch_k_fast'] = self.safe_float(fastk[current_idx])
            indicators['stoch_d_fast'] = self.safe_float(fastd[current_idx])
        else:
            indicators['stoch_k_fast'] = None
            indicators['stoch_d_fast'] = None

        # Stochastic - Slow
        if len(high) >= 14:
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            indicators['stoch_k_slow'] = self.safe_float(slowk[current_idx])
            indicators['stoch_d_slow'] = self.safe_float(slowd[current_idx])
        else:
            indicators['stoch_k_slow'] = None
            indicators['stoch_d_slow'] = None

        # ATR
        if len(high) >= 14:
            indicators['atr_14'] = self.safe_float(talib.ATR(high, low, close, timeperiod=14)[current_idx])
        else:
            indicators['atr_14'] = None

        # Beta (requires benchmark data, only for daily frequency)
        if benchmark_df is not None and current_date in benchmark_df.index:
            indicators['beta_252'] = self.calculate_beta(df, benchmark_df, current_date)

        # High/Low (using rolling windows)
        indicators['high_1w'] = self.safe_float(df['high'].rolling(5).max().iloc[current_idx]) if len(df) >= 5 else None
        indicators['low_1w'] = self.safe_float(df['low'].rolling(5).min().iloc[current_idx]) if len(df) >= 5 else None
        indicators['high_1m'] = self.safe_float(df['high'].rolling(21).max().iloc[current_idx]) if len(df) >= 21 else None
        indicators['low_1m'] = self.safe_float(df['low'].rolling(21).min().iloc[current_idx]) if len(df) >= 21 else None
        indicators['high_3m'] = self.safe_float(df['high'].rolling(63).max().iloc[current_idx]) if len(df) >= 63 else None
        indicators['low_3m'] = self.safe_float(df['low'].rolling(63).min().iloc[current_idx]) if len(df) >= 63 else None
        indicators['high_1y'] = self.safe_float(df['high'].rolling(252).max().iloc[current_idx]) if len(df) >= 252 else None
        indicators['low_1y'] = self.safe_float(df['low'].rolling(252).min().iloc[current_idx]) if len(df) >= 252 else None

        # Performance metrics
        indicators['perf_week'] = self.return_over_period(close, current_idx, 5)
        indicators['perf_month'] = self.return_over_period(close, current_idx, 21)
        indicators['perf_quarter'] = self.return_over_period(close, current_idx, 63)
        indicators['perf_half_y'] = self.return_over_period(close, current_idx, 126)
        indicators['perf_year'] = self.return_over_period(close, current_idx, 252)

        # YTD performance
        year_start = pd.Timestamp(current_date.year, 1, 1).tz_localize('UTC')
        if year_start in df.index:
            year_start_idx = df.index.get_loc(year_start)
            indicators['perf_ytd'] = ((current_price - df['close'].iloc[year_start_idx]) /
                                      df['close'].iloc[year_start_idx]) * 100

        # Volume metrics (only if we have volume data)
        if volume is not None:
            indicators['volume'] = volume[current_idx] if len(volume) > 0 else None
            if len(volume) >= 20:
                indicators['avg_volume_20d'] = volume[-20:].mean()
            else:
                indicators['avg_volume_20d'] = None

            if len(volume) >= 50:
                indicators['avg_volume_50d'] = volume[-50:].mean()
            else:
                indicators['avg_volume_50d'] = None

            if indicators.get('avg_volume_20d') and indicators.get('avg_volume_20d') > 0:
                indicators['relative_volume'] = volume[current_idx] / indicators['avg_volume_20d']
            else:
                indicators['relative_volume'] = None

            # VWAP calculations (only if we have volume)
            indicators['vwap_week'] = self.calculate_vwap(df[-5:]) if len(df) >= 5 else None
            indicators['vwap_month'] = self.calculate_vwap(df[-21:]) if len(df) >= 21 else None
        else:
            indicators['volume'] = None
            indicators['avg_volume_20d'] = None
            indicators['avg_volume_50d'] = None
            indicators['relative_volume'] = None
            indicators['vwap_week'] = None
            indicators['vwap_month'] = None

        # POC calculations (use price-based method when no volume)
        indicators['poc_week'] = self.calculate_poc(df[-5:], has_volume) if len(df) >= 5 else None
        indicators['poc_month'] = self.calculate_poc(df[-21:], has_volume) if len(df) >= 21 else None

        # Change from open and gap
        if open_prices[current_idx] > 0:
            indicators['change_open'] = ((close[current_idx] - open_prices[current_idx]) /
                                         open_prices[current_idx]) * 100
        if current_idx > 0 and close[current_idx - 1] > 0:
            prev_close = close[current_idx - 1]
            indicators['gap'] = ((open_prices[current_idx] - prev_close) / prev_close) * 100

        # 52-week range percentage
        if indicators.get('high_1y') and indicators.get('low_1y'):
            range_52w = indicators['high_1y'] - indicators['low_1y']
            if range_52w > 0:
                indicators['range_52w'] = ((current_price - indicators['low_1y']) / range_52w) * 100

        return indicators

    def safe_float(self, value):
        """Convert to float if not NaN"""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def return_over_period(self, prices, current_idx, period):
        """Calculate return over a period"""
        if current_idx >= period and prices[current_idx - period] > 0:
            prev_price = prices[current_idx - period]
            if prev_price > 0:
                return ((prices[current_idx] - prev_price) / prev_price) * 100
        return None

    def calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        if len(df) == 0:
            return None

        # Filter out zero volume rows
        df_filtered = df[df['volume'] > 0].copy()
        if len(df_filtered) == 0:
            return None

        typical_price = (df_filtered['high'] + df_filtered['low'] + df_filtered['close']) / 3
        total_volume = df_filtered['volume'].sum()

        if total_volume > 0:
            vwap = (typical_price * df_filtered['volume']).sum() / total_volume
            return float(vwap)
        return None

    def calculate_poc(self, df, has_volume=True):
        """Calculate Point of Control (price level with highest volume)"""
        if len(df) == 0:
            return None

        if has_volume and 'volume' in df.columns and df['volume'].sum() > 0:
            # Filter out zero volume rows
            df_filtered = df[df['volume'] > 0].copy()
            if len(df_filtered) > 0:
                # Use median as a simple proxy
                return float(df_filtered['close'].median())

        # Fallback to median price
        return float(df['close'].median())

    def calculate_beta(self, stock_df, benchmark_df, current_date):
        """Calculate beta vs benchmark (e.g., SPY) over last 252 days"""
        try:
            # Get stock returns for last 252 days
            stock_returns = stock_df['close'].pct_change().dropna().tail(252)

            # Get benchmark returns for same period
            if current_date in benchmark_df.index:
                current_idx = benchmark_df.index.get_loc(current_date)
                benchmark_returns = benchmark_df['close'].pct_change().dropna().iloc[max(0, current_idx-252):current_idx]

                if len(stock_returns) > 0 and len(benchmark_returns) > 0:
                    # Align the series
                    min_len = min(len(stock_returns), len(benchmark_returns))
                    stock_returns = stock_returns.iloc[-min_len:]
                    benchmark_returns = benchmark_returns.iloc[-min_len:]

                    covariance = np.cov(stock_returns, benchmark_returns)[0, 1]
                    variance = np.var(benchmark_returns)
                    if variance > 0:
                        return covariance / variance
        except Exception as e:
            logger.debug(f"Beta calculation error: {e}")
            pass
        return None

    def get_benchmark_data(self):
        """Get S&P 500 (SPY) data for beta calculation"""
        try:
            spy = Symbol.objects.get(ticker__in=['SPY', 'SP500'])
            ohlcv_data = OHLCVData.objects.filter(
                symbol=spy,
                frequency='1D'
            ).order_by('year')

            dfs = []
            for data in ohlcv_data:
                df = data.get_data_as_dataframe()
                if df is not None and not df.empty:
                    # Filter out zero volume days
                    if 'volume' in df.columns:
                        df = df[df['volume'] > 0].copy()
                    if not df.empty:
                        dfs.append(df)

            import ipdb;ipdb.set_trace()
            if dfs:
                benchmark = pd.concat(dfs).sort_index()
                self.stdout.write(f"  Loaded benchmark data: {len(benchmark)} days, {benchmark.index[0].date()} to {benchmark.index[-1].date()}")
                return benchmark
            else:
                self.stdout.write(self.style.WARNING("  SPY data found but no valid OHLCV data"))
                return None
        except Symbol.DoesNotExist:
            self.stdout.write(self.style.WARNING("  SPY symbol not found for beta calculation"))
            return None

