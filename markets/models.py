from django.db import models
from django.contrib.postgres.fields import JSONField
from django.core.validators import MinValueValidator, MaxValueValidator
import json

class Sector(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name

class Industry(models.Model):
    name = models.CharField(max_length=100)
    sector = models.ForeignKey(Sector, on_delete=models.CASCADE, related_name='industries')
    description = models.TextField(blank=True)

    class Meta:
        ordering = ['name']
        unique_together = ['name', 'sector']

    def __str__(self):
        return f"{self.name} ({self.sector.name})"

class Symbol(models.Model):
    FREQUENCY_CHOICES = [
        ('1min', '1 Minute'),
        ('5min', '5 Minutes'),
        ('15min', '15 Minutes'),
        ('1H', '1 Hour'),
        ('4H', '4 Hours'),
        ('1D', '1 Day'),
    ]

    ticker = models.CharField(max_length=20, db_index=True)
    name = models.CharField(max_length=200, blank=True)
    sector = models.ForeignKey(Sector, on_delete=models.SET_NULL, null=True, blank=True)
    industry = models.ForeignKey(Industry, on_delete=models.SET_NULL, null=True, blank=True)
    market_cap = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['ticker']
        indexes = [
            models.Index(fields=['ticker', 'is_active']),
        ]

    def __str__(self):
        return self.ticker

class OHLCVData(models.Model):
    """
    Stores OHLCV data in a JSON field by year and frequency for efficiency
    """
    FREQUENCY_CHOICES = [
        ('1min', '1 Minute'),
        ('5min', '5 Minutes'),
        ('15min', '15 Minutes'),
        ('1H', '1 Hour'),
        ('4H', '4 Hours'),
        ('1D', '1 Day'),
    ]

    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='ohlcv_data')
    frequency = models.CharField(max_length=10, choices=FREQUENCY_CHOICES, db_index=True)
    year = models.PositiveSmallIntegerField(db_index=True)

    # JSON field containing OHLCV data for the entire year
    # Structure: {
    #   "dates": ["2026-01-02", "2026-01-03", ...],
    #   "open": [100.5, 101.2, ...],
    #   "high": [102.3, 103.1, ...],
    #   "low": [99.8, 100.5, ...],
    #   "close": [101.1, 102.4, ...],
    #   "volume": [1000000, 1200000, ...]
    # }
    data = models.JSONField()

    # Metadata
    first_date = models.DateField(null=True, blank=True)
    last_date = models.DateField(null=True, blank=True)
    total_records = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['symbol', 'frequency', 'year']
        unique_together = ['symbol', 'frequency', 'year']
        indexes = [
            models.Index(fields=['symbol', 'frequency', 'year']),
            models.Index(fields=['frequency', 'year']),
        ]

    def __str__(self):
        return f"{self.symbol.ticker} - {self.frequency} - {self.year}"

    def get_data_as_dataframe(self):
        """Convert stored data to pandas DataFrame with proper date handling"""
        import pandas as pd
        import numpy as np

        # Create DataFrame from the stored data
        df = pd.DataFrame(self.data)

        if 'dates' in df.columns:
            # Convert dates with explicit format and UTC
            df['dates'] = pd.to_datetime(df['dates'], utc=True)
            df.set_index('dates', inplace=True)

            # Ensure all numeric columns are float
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        return df


class PrecomputedMetrics(models.Model):
    """
    Precomputed percentage changes for faster querying
    """
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='metrics')
    frequency = models.CharField(max_length=10, choices=OHLCVData.FREQUENCY_CHOICES)
    as_of_date = models.DateField(db_index=True)

    # Historical returns (backward-looking)
    change_1d = models.FloatField(null=True, blank=True)
    change_1w = models.FloatField(null=True, blank=True)
    change_2w = models.FloatField(null=True, blank=True)
    change_1m = models.FloatField(null=True, blank=True)
    change_3m = models.FloatField(null=True, blank=True)
    change_6m = models.FloatField(null=True, blank=True)
    change_1y = models.FloatField(null=True, blank=True)

    # Forward-looking metrics
    fwd_max_rise_1d = models.FloatField(null=True, blank=True, help_text="Maximum % rise in next 1 day")
    fwd_max_drop_1d = models.FloatField(null=True, blank=True, help_text="Maximum % drop in next 1 day")

    fwd_max_rise_1w = models.FloatField(null=True, blank=True, help_text="Maximum % rise in next 1 week")
    fwd_max_drop_1w = models.FloatField(null=True, blank=True, help_text="Maximum % drop in next 1 week")

    fwd_max_rise_2w = models.FloatField(null=True, blank=True, help_text="Maximum % rise in next 2 weeks")
    fwd_max_drop_2w = models.FloatField(null=True, blank=True, help_text="Maximum % drop in next 2 weeks")

    fwd_max_rise_1m = models.FloatField(null=True, blank=True, help_text="Maximum % rise in next 1 month")
    fwd_max_drop_1m = models.FloatField(null=True, blank=True, help_text="Maximum % drop in next 1 month")

    fwd_max_rise_3m = models.FloatField(null=True, blank=True, help_text="Maximum % rise in next 3 months")
    fwd_max_drop_3m = models.FloatField(null=True, blank=True, help_text="Maximum % drop in next 3 months")

    fwd_max_rise_6m = models.FloatField(null=True, blank=True, help_text="Maximum % rise in next 6 months")
    fwd_max_drop_6m = models.FloatField(null=True, blank=True, help_text="Maximum % drop in next 6 months")

    fwd_max_rise_1y = models.FloatField(null=True, blank=True, help_text="Maximum % rise in next 1 year")
    fwd_max_drop_1y = models.FloatField(null=True, blank=True, help_text="Maximum % drop in next 1 year")

    # Additional useful metrics
    fwd_volatility_1m = models.FloatField(null=True, blank=True, help_text="Volatility (std dev of returns) over next 1 month")
    fwd_volatility_3m = models.FloatField(null=True, blank=True, help_text="Volatility over next 3 months")
    fwd_volatility_6m = models.FloatField(null=True, blank=True, help_text="Volatility over next 6 months")

    fwd_sharpe_ratio = models.FloatField(null=True, blank=True, help_text="Sharpe ratio over next year (assuming risk-free rate of 2%)")

    fwd_max_drawdown = models.FloatField(null=True, blank=True, help_text="Maximum drawdown over next year")
    fwd_drawdown_duration = models.IntegerField(null=True, blank=True, help_text="Longest drawdown duration in days over next year")

    # Current price (keep as is)
    current_price = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-as_of_date', 'symbol']
        unique_together = ['symbol', 'frequency', 'as_of_date']
        indexes = [
            models.Index(fields=['symbol', 'frequency', '-as_of_date']),
        ]

    def __str__(self):
        return f"{self.symbol.ticker} - {self.as_of_date}"

