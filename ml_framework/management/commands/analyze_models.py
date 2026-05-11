from django.core.management.base import BaseCommand
from ml_framework.models import MLModel, BacktestResult
from markets.models import Symbol
import pandas as pd
from tabulate import tabulate

class Command(BaseCommand):
    help = 'Analyze and compare trained models'

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, help='Filter by symbol')
        parser.add_argument('--min-accuracy', type=float, default=0.5, help='Minimum accuracy threshold')

    def handle(self, *args, **options):
        symbol_filter = options.get('symbol')
        min_accuracy = options.get('min_accuracy')

        self.stdout.write("\n" + "="*80)
        self.stdout.write("MODEL PERFORMANCE ANALYSIS")
        self.stdout.write("="*80)

        # Get all models
        models = MLModel.objects.filter(is_active=True)
        if symbol_filter:
            models = models.filter(dataset__symbol__ticker=symbol_filter)

        if not models:
            self.stdout.write(self.style.ERROR("No models found"))
            return

        # Prepare data for comparison
        model_data = []
        for model in models:
            metrics = model.performance_metrics
            accuracy = metrics.get('accuracy', 0)

            if accuracy < min_accuracy:
                continue

            # Get best backtest if available
            best_backtest = model.backtests.order_by('-sharpe_ratio').first()

            model_data.append({
                'ID': model.id,
                'Name': model.name,
                'Algorithm': model.algorithm,
                'Symbol': model.dataset.symbol.ticker if model.dataset.symbol else 'N/A',
                'Accuracy': f"{accuracy:.2%}",
                'Precision': f"{metrics.get('precision', 0):.2%}",
                'Recall': f"{metrics.get('recall', 0):.2%}",
                'F1': f"{metrics.get('f1_score', 0):.2%}",
                'ROC-AUC': f"{metrics.get('roc_auc', 0):.2%}",
                'Sharpe': f"{best_backtest.sharpe_ratio:.2f}" if best_backtest else 'N/A',
                'Win Rate': f"{best_backtest.win_rate:.2%}" if best_backtest else 'N/A',
                'Trades': best_backtest.total_trades if best_backtest else 0,
                'Training Date': model.training_date
            })

        if not model_data:
            self.stdout.write(self.style.WARNING(f"No models with accuracy > {min_accuracy:.0%}"))
            return

        # Create DataFrame for display
        df = pd.DataFrame(model_data)

        # Sort by accuracy and Sharpe ratio
        df_sorted = df.sort_values(['Accuracy', 'Sharpe'], ascending=False)

        self.stdout.write("\n" + tabulate(df_sorted, headers='keys', tablefmt='grid', showindex=False))

        # Show best model
        best_model = df_sorted.iloc[0]
        self.stdout.write("\n" + "="*80)
        self.stdout.write(f"🏆 BEST MODEL: {best_model['Name']}")
        self.stdout.write("="*80)
        self.stdout.write(f"Symbol: {best_model['Symbol']}")
        self.stdout.write(f"Accuracy: {best_model['Accuracy']}")
        self.stdout.write(f"Sharpe Ratio: {best_model['Sharpe']}")
        self.stdout.write(f"Win Rate: {best_model['Win Rate']}")

        # Recommendations
        self.stdout.write("\n" + "="*80)
        self.stdout.write("RECOMMENDATIONS")
        self.stdout.write("="*80)

        best_accuracy = float(best_model['Accuracy'].strip('%')) / 100

        if best_accuracy > 0.55:
            self.stdout.write(self.style.SUCCESS("✓ Model shows promising performance!"))
            self.stdout.write("  Next steps:")
            self.stdout.write("  1. Paper trade with this model")
            self.stdout.write("  2. Set up daily predictions")
            self.stdout.write("  3. Monitor real-world performance")
        elif best_accuracy > 0.52:
            self.stdout.write(self.style.WARNING("⚠ Model slightly better than random"))
            self.stdout.write("  Next steps:")
            self.stdout.write("  1. Test with out-of-sample data")
            self.stdout.write("  2. Add more features (fundamental data)")
            self.stdout.write("  3. Try ensemble methods")
        else:
            self.stdout.write(self.style.ERROR("❌ Model performance is near random"))
            self.stdout.write("  Improvement strategies:")
            self.stdout.write("  1. Use longer lookahead periods (1-3 months)")
            self.stdout.write("  2. Add fundamental data (P/E, earnings, sentiment)")
            self.stdout.write("  3. Try different frequencies (1H, 4H)")
            self.stdout.write("  4. Use feature selection to reduce noise")
            self.stdout.write("  5. Consider regression instead of classification")
