from django.core.management.base import BaseCommand
from markets.models import PrecomputedMetrics, Symbol
from django.db.models import Count, Q
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Check and fix PrecomputedMetrics data'

    def add_arguments(self, parser):
        parser.add_argument('--fix', action='store_true', help='Fix missing data')
        parser.add_argument('--symbol', type=str, help='Check specific symbol')

    def handle(self, *args, **options):
        fix = options.get('fix', False)
        symbol_filter = options.get('symbol')

        # Get queryset
        queryset = PrecomputedMetrics.objects.filter(frequency='1D')
        if symbol_filter:
            queryset = queryset.filter(symbol__ticker=symbol_filter)

        # Check total records
        total_records = queryset.count()
        self.stdout.write(f"Total PrecomputedMetrics records: {total_records}")

        # Check for null values in each field
        fields_to_check = [
            'change_1d', 'change_1w', 'change_2w', 'change_1m',
            'change_3m', 'change_6m', 'change_1y', 'current_price'
        ]

        self.stdout.write("\n=== NULL VALUE COUNTS ===")
        for field in fields_to_check:
            null_count = queryset.filter(**{f"{field}__isnull": True}).count()
            self.stdout.write(f"{field}: {null_count} null out of {total_records} ({null_count/total_records*100:.1f}%)")

        # Check for zero values
        self.stdout.write("\n=== ZERO VALUE COUNTS ===")
        for field in fields_to_check:
            zero_count = queryset.filter(**{field: 0}).count()
            self.stdout.write(f"{field}: {zero_count} zero values")

        # Show sample of records with missing data
        self.stdout.write("\n=== SAMPLE RECORDS WITH MISSING DATA ===")
        missing_data = queryset.filter(
            Q(change_1d__isnull=True) |
            Q(change_1w__isnull=True) |
            Q(change_1m__isnull=True) |
            Q(change_1y__isnull=True)
        )[:5]

        for record in missing_data:
            self.stdout.write(f"\nSymbol: {record.symbol.ticker}")
            self.stdout.write(f"  As of Date: {record.as_of_date}")
            self.stdout.write(f"  change_1d: {record.change_1d}")
            self.stdout.write(f"  change_1w: {record.change_1w}")
            self.stdout.write(f"  change_1m: {record.change_1m}")
            self.stdout.write(f"  change_1y: {record.change_1y}")

        # Show sample of complete records
        self.stdout.write("\n=== SAMPLE COMPLETE RECORDS ===")
        complete_records = queryset.exclude(
            Q(change_1d__isnull=True) |
            Q(change_1w__isnull=True) |
            Q(change_1m__isnull=True) |
            Q(change_1y__isnull=True)
        )[:5]

        for record in complete_records:
            self.stdout.write(f"\nSymbol: {record.symbol.ticker}")
            self.stdout.write(f"  As of Date: {record.as_of_date}")
            self.stdout.write(f"  current_price: {record.current_price}")
            self.stdout.write(f"  change_1d: {record.change_1d}%")
            self.stdout.write(f"  change_1w: {record.change_1w}%")
            self.stdout.write(f"  change_1m: {record.change_1m}%")
            self.stdout.write(f"  change_1y: {record.change_1y}%")

