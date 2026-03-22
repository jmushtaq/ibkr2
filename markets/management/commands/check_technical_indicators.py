# markets/management/commands/check_technical_indicators.py

"""
Management command to check and validate technical indicators data.

Usage:
    python manage.py check_technical_indicators
    python manage.py check_technical_indicators --symbol AAPL
    python manage.py check_technical_indicators --symbol SP500 --verbose
    python manage.py check_technical_indicators --all-symbols --detailed
    python manage.py check_technical_indicators --fix-missing

    # Check basic stats for first 10 symbols
    python manage.py check_technical_indicators

    # Check specific symbol with detailed output
    python manage.py check_technical_indicators --symbol SP500 --detailed

    # Check all symbols with quality checks
    python manage.py check_technical_indicators --all-symbols --quality-check

    # Show gaps in data
    python manage.py check_technical_indicators --symbol AAPL --show-gaps

    # Complete check with all options
    python manage.py check_technical_indicators --symbol AAPL --verbose --detailed --quality-check --show-gaps

    # Check all symbols with detailed quality checks
    python manage.py check_technical_indicators --all-symbols --quality-check --detailed
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.db.models import Count, Avg, Min, Max, Q
from markets.models import Symbol, OHLCVData, PrecomputedMetrics, TechnicalIndicators
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Check and validate technical indicators data'

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, help='Check specific symbol')
        parser.add_argument('--all-symbols', action='store_true', help='Check all symbols')
        parser.add_argument('--verbose', action='store_true', help='Show detailed output')
        parser.add_argument('--detailed', action='store_true', help='Show even more details')
        parser.add_argument('--fix-missing', action='store_true', help='Identify missing data')
        parser.add_argument('--show-gaps', action='store_true', help='Show data gaps')
        parser.add_argument('--quality-check', action='store_true', help='Run quality checks on indicators')

    def handle(self, *args, **options):
        symbol_filter = options.get('symbol')
        all_symbols = options.get('all_symbols')
        verbose = options.get('verbose', False)
        detailed = options.get('detailed', False)
        fix_missing = options.get('fix_missing', False)
        show_gaps = options.get('show_gaps', False)
        quality_check = options.get('quality_check', False)

        if symbol_filter:
            symbols = Symbol.objects.filter(ticker=symbol_filter)
        elif all_symbols:
            symbols = Symbol.objects.filter(is_active=True).order_by('ticker')
        else:
            symbols = Symbol.objects.filter(is_active=True).order_by('ticker')[:10]
            self.stdout.write(self.style.WARNING(
                "Showing first 10 symbols. Use --all-symbols to see all, or --symbol to check specific."
            ))

        self.stdout.write(f"\n{'='*80}")
        self.stdout.write(f"TECHNICAL INDICATORS VALIDATION REPORT")
        self.stdout.write(f"{'='*80}\n")

        total_checks = 0
        symbols_with_issues = 0

        for symbol in symbols:
            self.stdout.write(f"\n📊 {symbol.ticker}")
            self.stdout.write("-" * 40)

            # Check PrecomputedMetrics
            metrics_count = PrecomputedMetrics.objects.filter(
                symbol=symbol,
                frequency='1D'
            ).count()

            # Check TechnicalIndicators
            indicators_count = TechnicalIndicators.objects.filter(
                symbol=symbol,
                frequency='1D'
            ).count()

            # Get date ranges
            metrics_dates = PrecomputedMetrics.objects.filter(
                symbol=symbol,
                frequency='1D'
            ).aggregate(
                first=Min('as_of_date'),
                last=Max('as_of_date')
            )

            indicators_dates = TechnicalIndicators.objects.filter(
                symbol=symbol,
                frequency='1D'
            ).aggregate(
                first=Min('as_of_date'),
                last=Max('as_of_date')
            )

            # Get OHLCV data info
            ohlcv_data = OHLCVData.objects.filter(
                symbol=symbol,
                frequency='1D'
            ).order_by('year')

            ohlcv_days = 0
            first_ohlcv = None
            last_ohlcv = None

            if ohlcv_data.exists():
                dfs = []
                for data in ohlcv_data:
                    df = data.get_data_as_dataframe()
                    if df is not None and not df.empty:
                        dfs.append(df)
                if dfs:
                    combined = pd.concat(dfs)
                    combined.sort_index(inplace=True)
                    ohlcv_days = len(combined)
                    first_ohlcv = combined.index[0].date()
                    last_ohlcv = combined.index[-1].date()

            # Basic stats
            self.stdout.write(f"  📈 PrecomputedMetrics: {metrics_count} records")
            self.stdout.write(f"     Date range: {metrics_dates['first']} to {metrics_dates['last']}" if metrics_dates['first'] else "     No data")
            self.stdout.write(f"  🔧 TechnicalIndicators: {indicators_count} records")
            self.stdout.write(f"     Date range: {indicators_dates['first']} to {indicators_dates['last']}" if indicators_dates['first'] else "     No data")
            self.stdout.write(f"  📊 OHLCV data: {ohlcv_days} days")
            if first_ohlcv and last_ohlcv:
                self.stdout.write(f"     Date range: {first_ohlcv} to {last_ohlcv}")

            # Coverage analysis
            if metrics_dates['first'] and indicators_dates['first']:
                coverage = (indicators_count / metrics_count * 100) if metrics_count > 0 else 0
                self.stdout.write(f"  📊 Coverage: {coverage:.1f}% ({indicators_count}/{metrics_count})")

                # Check if we have the expected 200-day buffer
                if first_ohlcv and indicators_dates['first']:
                    expected_start = first_ohlcv + timedelta(days=200)
                    actual_start = indicators_dates['first']
                    days_diff = (actual_start - expected_start).days if actual_start else 0

                    if actual_start >= expected_start:
                        self.stdout.write(self.style.SUCCESS(
                            f"  ✅ 200-day buffer satisfied: indicators start {actual_start} "
                            f"(expected {expected_start}, diff +{days_diff} days)"
                        ))
                    else:
                        self.stdout.write(self.style.WARNING(
                            f"  ⚠️ 200-day buffer not fully satisfied: indicators start {actual_start} "
                            f"(expected {expected_start}, diff {days_diff} days)"
                        ))

            if verbose or detailed:
                # Show null value counts for key indicators
                null_counts = TechnicalIndicators.objects.filter(
                    symbol=symbol,
                    frequency='1D'
                ).aggregate(
                    rsi_null=Count('rsi_14', filter=Q(rsi_14__isnull=True)),
                    sma50_null=Count('sma_50', filter=Q(sma_50__isnull=True)),
                    sma200_null=Count('sma_200', filter=Q(sma_200__isnull=True)),
                    atr_null=Count('atr_14', filter=Q(atr_14__isnull=True)),
                    beta_null=Count('beta_252', filter=Q(beta_252__isnull=True)),
                    volume_null=Count('volume', filter=Q(volume__isnull=True)),
                )

                self.stdout.write(f"\n  🔍 Null value counts:")
                self.stdout.write(f"     RSI 14: {null_counts['rsi_null']}")
                self.stdout.write(f"     SMA 50: {null_counts['sma50_null']}")
                self.stdout.write(f"     SMA 200: {null_counts['sma200_null']}")
                self.stdout.write(f"     ATR 14: {null_counts['atr_null']}")
                self.stdout.write(f"     Beta 252: {null_counts['beta_null']}")
                self.stdout.write(f"     Volume: {null_counts['volume_null']}")

            if detailed:
                # Show sample of recent data
                recent = TechnicalIndicators.objects.filter(
                    symbol=symbol,
                    frequency='1D'
                ).order_by('-as_of_date')[:5]

                if recent.exists():
                    self.stdout.write(f"\n  📋 Recent indicators (last 5 days):")
                    for r in recent:
                        rsi_str = f"{r.rsi_14:.1f}" if r.rsi_14 is not None else "N/A"
                        sma50_str = f"{r.sma_50:.2f}" if r.sma_50 is not None else "N/A"
                        sma200_str = f"{r.sma_200:.2f}" if r.sma_200 is not None else "N/A"
                        self.stdout.write(f"     {r.as_of_date}: RSI={rsi_str}, SMA50={sma50_str}, SMA200={sma200_str}")

            if show_gaps:
                # Check for gaps in TechnicalIndicators
                indicators = TechnicalIndicators.objects.filter(
                    symbol=symbol,
                    frequency='1D'
                ).order_by('as_of_date').values_list('as_of_date', flat=True)

                if len(indicators) > 1:
                    dates_list = list(indicators)
                    gaps = []
                    for i in range(1, len(dates_list)):
                        days_diff = (dates_list[i] - dates_list[i-1]).days
                        if days_diff > 5:  # More than 5 days gap
                            gaps.append((dates_list[i-1], dates_list[i], days_diff))

                    if gaps:
                        self.stdout.write(self.style.WARNING(f"\n  ⚠️ Gaps found in indicators:"))
                        for gap_start, gap_end, days in gaps[:5]:
                            self.stdout.write(f"     {gap_start} → {gap_end}: {days} days")
                    else:
                        self.stdout.write(f"\n  ✅ No significant gaps found")

            if quality_check:
                # Run quality checks on indicator values
                self.run_quality_checks(symbol)

            total_checks += 1

        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*80}\n"
            f"VALIDATION COMPLETE\n"
            f"{'='*80}\n"
            f"Symbols checked: {total_checks}\n"
        ))

    def run_quality_checks(self, symbol):
        """Run quality checks on technical indicators"""
        self.stdout.write(f"\n  🔬 Quality Checks:")

        # Check for out-of-range values
        out_of_range = TechnicalIndicators.objects.filter(
            symbol=symbol,
            frequency='1D'
        ).filter(
            Q(rsi_14__lt=0) | Q(rsi_14__gt=100) |
            Q(stoch_k_fast__lt=0) | Q(stoch_k_fast__gt=100) |
            Q(stoch_d_fast__lt=0) | Q(stoch_d_fast__gt=100) |
            Q(stoch_k_slow__lt=0) | Q(stoch_k_slow__gt=100) |
            Q(stoch_d_slow__lt=0) | Q(stoch_d_slow__gt=100)
        ).count()

        if out_of_range > 0:
            self.stdout.write(self.style.WARNING(
                f"     ⚠️ Out-of-range values: {out_of_range} records"
            ))

        # Check for negative moving averages (shouldn't happen for prices)
        negative_ma = TechnicalIndicators.objects.filter(
            symbol=symbol,
            frequency='1D'
        ).filter(
            Q(sma_50__lt=0) | Q(sma_200__lt=0) | Q(ema_50__lt=0)
        ).count()

        if negative_ma > 0:
            self.stdout.write(self.style.WARNING(
                f"     ⚠️ Negative moving averages: {negative_ma} records"
            ))

        # Check for reasonable RSI distribution
        rsi_values = TechnicalIndicators.objects.filter(
            symbol=symbol,
            frequency='1D',
            rsi_14__isnull=False
        ).values_list('rsi_14', flat=True)

        if rsi_values:
            rsi_list = list(rsi_values)
            overbought = sum(1 for r in rsi_list if r > 70)
            oversold = sum(1 for r in rsi_list if r < 30)
            normal = len(rsi_list) - overbought - oversold

            self.stdout.write(f"     RSI Distribution: Overbought (>70): {overbought}, "
                             f"Oversold (<30): {oversold}, Normal: {normal}")

            if overbought + oversold > 0:
                pct_extreme = (overbought + oversold) / len(rsi_list) * 100
                self.stdout.write(f"     Extreme RSI conditions: {pct_extreme:.1f}% of days")

        # Check SMA 50 vs SMA 200 relationship (Golden Cross/Death Cross)
        sma_cross = TechnicalIndicators.objects.filter(
            symbol=symbol,
            frequency='1D',
            sma_50__isnull=False,
            sma_200__isnull=False
        ).order_by('as_of_date').values_list('as_of_date', 'sma_50', 'sma_200')

        if sma_cross:
            crosses = 0
            prev_above = None
            for date, sma50, sma200 in sma_cross:
                current_above = sma50 > sma200
                if prev_above is not None and current_above != prev_above:
                    crosses += 1
                    cross_type = "Golden Cross" if current_above else "Death Cross"
                    self.stdout.write(f"     {date}: {cross_type} (SMA50: {sma50:.2f}, SMA200: {sma200:.2f})")
                prev_above = current_above

            if crosses > 0:
                self.stdout.write(f"     Total crossovers detected: {crosses}")

        # Check for consistent ATR (should be positive)
        zero_atr = TechnicalIndicators.objects.filter(
            symbol=symbol,
            frequency='1D',
            atr_14__isnull=False,
            atr_14=0
        ).count()

        if zero_atr > 0:
            self.stdout.write(self.style.WARNING(
                f"     ⚠️ Zero ATR values: {zero_atr} records (possible data issues)"
            ))

        # Volume check (if applicable)
        has_volume = TechnicalIndicators.objects.filter(
            symbol=symbol,
            frequency='1D',
            volume__isnull=False,
            volume__gt=0
        ).exists()

        if has_volume:
            zero_volume = TechnicalIndicators.objects.filter(
                symbol=symbol,
                frequency='1D',
                volume=0
            ).count()
            if zero_volume > 0:
                self.stdout.write(self.style.WARNING(
                    f"     ⚠️ Zero volume days: {zero_volume} records"
                ))

    def check_data_quality(self, symbol):
        """Additional data quality checks"""
        self.stdout.write(f"\n  📊 Data Quality:")

        # Get latest indicators
        latest = TechnicalIndicators.objects.filter(
            symbol=symbol,
            frequency='1D'
        ).order_by('-as_of_date').first()

        if latest:
            self.stdout.write(f"     Latest date: {latest.as_of_date}")
            if latest.rsi_14:
                rsi_status = "Overbought" if latest.rsi_14 > 70 else "Oversold" if latest.rsi_14 < 30 else "Neutral"
                self.stdout.write(f"     Current RSI: {latest.rsi_14:.1f} ({rsi_status})")

            if latest.sma_50 and latest.sma_200:
                trend = "Bullish" if latest.sma_50 > latest.sma_200 else "Bearish"
                self.stdout.write(f"     Trend: {trend} (SMA50 {'>' if latest.sma_50 > latest.sma_200 else '<'} SMA200)")

            if latest.beta_252:
                beta_status = "High Beta" if latest.beta_252 > 1.5 else "Low Beta" if latest.beta_252 < 0.5 else "Market Beta"
                self.stdout.write(f"     Beta: {latest.beta_252:.2f} ({beta_status})")

        # Check data freshness
        if latest and latest.as_of_date < datetime.now().date() - timedelta(days=5):
            self.stdout.write(self.style.WARNING(
                f"     ⚠️ Data may be stale (last update: {latest.as_of_date})"
            ))
