from django.core.management.base import BaseCommand
from markets.models import OHLCVData
import json

class Command(BaseCommand):
    help = 'Debug date storage in OHLCVData'

    def add_arguments(self, parser):
        parser.add_argument('--ticker', type=str, help='Ticker to debug')
        parser.add_argument('--frequency', type=str, default='1D', help='Frequency')
        parser.add_argument('--year', type=int, help='Year')

    def handle(self, *args, **options):
        ticker = options.get('ticker', 'FMCC')
        frequency = options.get('frequency', '1D')
        year = options.get('year', 2026)

        # Get the data
        queryset = OHLCVData.objects.filter(
            symbol__ticker=ticker,
            frequency=frequency,
            year=year
        )

        if not queryset.exists():
            self.stdout.write(self.style.ERROR(f"No data found for {ticker} {frequency} {year}"))
            return

        for data in queryset:
            self.stdout.write(f"\n{data.symbol.ticker} - {data.frequency} - {data.year}")
            self.stdout.write(f"First date in metadata: {data.first_date}")
            self.stdout.write(f"Last date in metadata: {data.last_date}")
            self.stdout.write(f"Total records: {data.total_records}")

            # Check the stored dates
            if 'dates' in data.data:
                dates = data.data['dates']
                self.stdout.write(f"\nFirst 5 stored dates: {dates[:5]}")
                self.stdout.write(f"Last 5 stored dates: {dates[-5:]}")

                # Check if dates are valid
                from datetime import datetime
                valid_dates = 0
                for date_str in dates[:10]:  # Check first 10
                    try:
                        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        self.stdout.write(f"  ✓ {date_str} -> {dt}")
                        valid_dates += 1
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f"  ✗ {date_str} -> {e}"))

                if valid_dates == 0:
                    self.stdout.write(self.style.ERROR("No valid dates found in stored data!"))
            else:
                self.stdout.write(self.style.ERROR("No 'dates' field in data!"))

