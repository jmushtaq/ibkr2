from django.core.management.base import BaseCommand
from ml_framework.models import MLModel
from ml_framework.backtest import Backtester
import pandas as pd
import numpy as np

class Command(BaseCommand):
    help = 'Analyze the best performing model in detail'

    def add_arguments(self, parser):
        parser.add_argument('--model-id', type=int, help='Specific model ID')
        parser.add_argument('--symbol', type=str, help='Symbol to analyze')

    def handle(self, *args, **options):
        # Get the best model
        if options['model_id']:
            model = MLModel.objects.get(id=options['model_id'])
        elif options['symbol']:
            model = MLModel.objects.filter(
                dataset__symbol__ticker=options['symbol'],
                is_active=True
            ).order_by('-performance_metrics__accuracy').first()
        else:
            model = MLModel.objects.filter(is_active=True).order_by('-performance_metrics__accuracy').first()

        if not model:
            self.stdout.write(self.style.ERROR("No model found"))
            return

        self.stdout.write("\n" + "="*80)
        self.stdout.write(f"DETAILED ANALYSIS: {model.name}")
        self.stdout.write("="*80)

        # 1. Basic metrics
        metrics = model.performance_metrics
        self.stdout.write("\n📊 PERFORMANCE METRICS:")
        self.stdout.write(f"  Accuracy:  {metrics['accuracy']:.2%}")
        self.stdout.write(f"  Precision: {metrics.get('precision', 0):.2%}")
        self.stdout.write(f"  Recall:    {metrics.get('recall', 0):.2%}")
        self.stdout.write(f"  F1 Score:  {metrics.get('f1_score', 0):.2%}")
        if 'roc_auc' in metrics:
            self.stdout.write(f"  ROC-AUC:   {metrics['roc_auc']:.2%}")

        # 2. Feature importance (top 10)
        importance = model.feature_importance
        if importance:
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            self.stdout.write("\n🔝 TOP 10 IMPORTANT FEATURES:")
            for i, (feature, imp) in enumerate(sorted_importance, 1):
                self.stdout.write(f"  {i}. {feature}: {imp:.4f}")

        # 3. Backtest results
        backtests = model.backtests.order_by('-sharpe_ratio')
        if backtests.exists():
            self.stdout.write("\n💰 BACKTEST RESULTS:")
            for bt in backtests[:3]:
                self.stdout.write(f"\n  {bt.name}:")
                self.stdout.write(f"    Sharpe Ratio: {bt.sharpe_ratio:.2f}")
                self.stdout.write(f"    Total Return: {bt.total_return:.2%}")
                self.stdout.write(f"    Win Rate: {bt.win_rate:.2%}")
                self.stdout.write(f"    Max Drawdown: {bt.max_drawdown:.2%}")
                self.stdout.write(f"    Total Trades: {bt.total_trades}")

        # 4. Model configuration
        self.stdout.write("\n⚙️ MODEL CONFIGURATION:")
        self.stdout.write(f"  Algorithm: {model.algorithm}")
        self.stdout.write(f"  Training Date: {model.training_date}")
        self.stdout.write(f"  Training Samples: {model.training_samples}")
        self.stdout.write(f"  Features: {len(model.feature_importance)}")

        # 5. Recommendations
        self.stdout.write("\n💡 RECOMMENDATIONS:")
        if metrics['accuracy'] > 0.7:
            self.stdout.write(self.style.SUCCESS("  ✓ This model is ready for paper trading!"))
            self.stdout.write("  Recommended actions:")
            self.stdout.write("    1. Set up daily signal generation")
            self.stdout.write("    2. Start paper trading to validate real-world performance")
            self.stdout.write("    3. Monitor Sharpe ratio and drawdown closely")
            self.stdout.write("    4. Consider using with position sizing based on confidence")
        else:
            self.stdout.write(self.style.WARNING("  ⚠ Model needs improvement before live trading"))

        self.stdout.write("\n" + "="*80)

