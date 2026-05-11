from django.core.management.base import BaseCommand
from ml_trend.models import MLTrendModel
import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime, timedelta

class Command(BaseCommand):
    help = 'Compare all trained models with training and validation metrics'

    def add_arguments(self, parser):
        parser.add_argument('--model-id', type=int, help='Compare specific model ID')
        parser.add_argument('--min-precision', type=float, default=0.3, help='Minimum precision to display')
        parser.add_argument('--export-csv', type=str, help='Export to CSV file')
        parser.add_argument('--show-features', action='store_true', help='Show top features for best model')
        parser.add_argument('--show-thresholds', action='store_true', help='Show threshold analysis if available')

    def handle(self, *args, **options):
        model_id = options.get('model_id')
        min_precision = options.get('min_precision')
        export_csv = options.get('export_csv')
        show_features = options.get('show_features')
        show_thresholds = options.get('show_thresholds')

        # Get models
        if model_id:
            models = MLTrendModel.objects.filter(id=model_id, is_active=True)
        else:
            models = MLTrendModel.objects.filter(is_active=True).order_by('-created_at')

        if not models:
            self.stdout.write(self.style.ERROR("No models found"))
            return

        self.stdout.write("\n" + "="*120)
        self.stdout.write("ML TREND FOLLOWING MODEL COMPARISON")
        self.stdout.write("="*120)

        # Collect model data
        model_data = []

        for model in models:
            metrics = model.performance_metrics

            # Training metrics
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1_score', 0)
            roc_auc = metrics.get('roc_auc', 0)
            true_positives = metrics.get('true_positives', 0)
            false_positives = metrics.get('false_positives', 0)

            # Expected value from training
            rr = model.target_rr_ratio
            expected_training = (precision * rr) - ((1 - precision) * 1)

            # Look for validation results file
            validation_metrics = self._find_validation_results(model)

            # Parse validation metrics
            val_precision = 'N/A'
            val_expected = 'N/A'
            val_expected_num = 0
            val_profit = 'N/A'
            best_threshold = 'N/A'
            val_signals = 'N/A'

            if validation_metrics:
                # Parse precision
                if 'precision' in validation_metrics:
                    val_prec_str = validation_metrics['precision']
                    if isinstance(val_prec_str, str):
                        val_precision = val_prec_str
                    else:
                        val_precision = f"{val_prec_str:.1%}"

                # Parse expected value
                if 'expected_value' in validation_metrics:
                    exp_val = validation_metrics['expected_value']
                    if isinstance(exp_val, str):
                        # Extract number from string like "1.100R"
                        match = re.search(r'([\d.-]+)', exp_val)
                        if match:
                            val_expected_num = float(match.group(1))
                            val_expected = exp_val
                            val_profit = f"${val_expected_num * 100:.0f}"
                    else:
                        val_expected_num = float(exp_val)
                        val_expected = f"{val_expected_num:.2f}R"
                        val_profit = f"${val_expected_num * 100:.0f}"

                # Parse best threshold
                if 'best_threshold' in validation_metrics:
                    bt = validation_metrics['best_threshold']
                    if isinstance(bt, (int, float)):
                        best_threshold = f"{bt:.2f}"
                    else:
                        best_threshold = str(bt)

                # Parse total signals
                if 'total_signals' in validation_metrics:
                    val_signals = validation_metrics['total_signals']

            # Format training metrics
            train_prec_str = f"{precision:.1%}"
            train_f1_str = f"{f1:.1%}"
            train_expected_str = f"{expected_training:+.2f}R"
            train_profit_str = f"${expected_training * 100:.0f}"

            model_data.append({
                'ID': model.id,
                'Name': model.name[:40],
                'RR': f"{rr}:1",
                'Training Date': model.training_date.strftime('%Y-%m-%d'),
                'Samples': model.training_samples,
                'Train_Acc': f"{accuracy:.1%}",
                'Train_Prec': train_prec_str,
                'Train_Recall': f"{recall:.1%}",
                'Train_F1': train_f1_str,
                'Train_AUC': f"{roc_auc:.1%}",
                'Train_TP': true_positives,
                'Train_FP': false_positives,
                'Train_Expected': train_expected_str,
                'Train_Profit_$100': train_profit_str,
                'Val_Prec': val_precision,
                'Val_Expected': val_expected,
                'Val_Profit_$100': val_profit,
                'Val_Signals': val_signals,
                'Best_Threshold': best_threshold,
                'Train_Expected_Num': expected_training,
                'Val_Expected_Num': val_expected_num
            })

        # Create DataFrame
        df = pd.DataFrame(model_data)

        # Filter by precision
        train_prec = df['Train_Prec'].str.rstrip('%').astype(float) / 100
        df_filtered = df[train_prec >= min_precision]

        if len(df_filtered) == 0:
            self.stdout.write(self.style.WARNING(f"No models with precision >= {min_precision:.0%}"))
            df_filtered = df

        # Sort by expected profit
        df_filtered = df_filtered.sort_values('Train_Expected_Num', ascending=False)

        # Display results
        self.stdout.write("\n📊 MODEL PERFORMANCE SUMMARY")
        self.stdout.write("-"*120)

        # Print header
        self.stdout.write(f"{'ID':<5} {'Model Name':<40} {'RR':<6} {'Train Date':<12} {'Samples':<8} "
                         f"{'Train Prec':<11} {'Train F1':<9} {'Train Exp':<10} "
                         f"{'Val Prec':<10} {'Val Exp':<10} {'Best Thr':<9}")
        self.stdout.write("-"*120)

        for _, row in df_filtered.iterrows():
            # Color code based on profitability
            train_exp = row['Train_Expected_Num']
            val_exp = row['Val_Expected_Num'] if row['Val_Expected_Num'] != 0 else 0

            train_color = self.style.SUCCESS if train_exp > 0 else self.style.ERROR
            val_color = self.style.SUCCESS if val_exp > 0 else self.style.WARNING

            # Handle potentially long names
            name = row['Name'][:38] + '...' if len(row['Name']) > 40 else row['Name']

            self.stdout.write(
                f"{row['ID']:<5} {name:<40} {row['RR']:<6} {row['Training Date']:<12} {row['Samples']:<8} "
                f"{row['Train_Prec']:<11} {row['Train_F1']:<9} "
                f"{train_color(row['Train_Expected']):<10} "
                f"{val_color(row['Val_Prec']):<10} {val_color(row['Val_Expected']):<10} "
                f"{row['Best_Threshold']:<9}"
            )

        # Best Model Analysis
        if len(df_filtered) > 0:
            self.stdout.write("\n" + "="*120)
            self.stdout.write("🏆 BEST MODEL ANALYSIS")
            self.stdout.write("="*120)

            best_model = models.filter(id=df_filtered.iloc[0]['ID']).first()
            if best_model:
                self._analyze_best_model(best_model, show_features, show_thresholds)

        # Summary Statistics
        self.stdout.write("\n" + "="*120)
        self.stdout.write("📈 SUMMARY STATISTICS")
        self.stdout.write("="*120)

        profitable_models = df_filtered[df_filtered['Train_Expected_Num'] > 0]
        self.stdout.write(f"Total Models: {len(df_filtered)}")
        self.stdout.write(f"Profitable Models: {len(profitable_models)} ({len(profitable_models)/len(df_filtered)*100:.1f}%)")

        if len(profitable_models) > 0:
            best_profit = profitable_models.iloc[0]['Train_Expected_Num']
            avg_profit = profitable_models['Train_Expected_Num'].mean()
            self.stdout.write(f"Best Expected Profit: {best_profit:.2f}R")
            self.stdout.write(f"Average Expected Profit: {avg_profit:.2f}R")

        # Validation Statistics
        validated_models = df_filtered[df_filtered['Val_Prec'] != 'N/A']
        if len(validated_models) > 0:
            self.stdout.write(f"\nModels with Validation Data: {len(validated_models)}")
            val_profitable = validated_models[validated_models['Val_Expected_Num'] > 0]
            if len(val_profitable) > 0:
                self.stdout.write(f"Validated Profitable Models: {len(val_profitable)} ({len(val_profitable)/len(validated_models)*100:.1f}%)")

        # Recommendations
        self.stdout.write("\n" + "="*120)
        self.stdout.write("💡 RECOMMENDATIONS")
        self.stdout.write("="*120)

        if len(profitable_models) > 0:
            best = profitable_models.iloc[0]
            self.stdout.write(self.style.SUCCESS(f"\n✓ Recommended Model: ID {best['ID']}"))
            self.stdout.write(f"  Name: {best['Name']}")
            self.stdout.write(f"  RR Ratio: {best['RR']}")
            self.stdout.write(f"  Expected Profit: {best['Train_Expected']} per trade")
            self.stdout.write(f"  Expected Profit per $100 risked: {best['Train_Profit_$100']}")
            self.stdout.write(f"  Precision: {best['Train_Prec']}")
            self.stdout.write(f"  F1 Score: {best['Train_F1']}")

            if best['Val_Prec'] != 'N/A':
                self.stdout.write(f"\n  Validation Performance:")
                self.stdout.write(f"    Out-of-Sample Precision: {best['Val_Prec']}")
                self.stdout.write(f"    Out-of-Sample Expected Value: {best['Val_Expected']}")
                self.stdout.write(f"    Best Threshold: {best['Best_Threshold']}")

            self.stdout.write(f"\n  To use this model:")
            self.stdout.write(f"    python manage.py generate_trading_signals --model-id {best['ID']} --threshold {best['Best_Threshold']}")
        else:
            self.stdout.write(self.style.WARNING("\n⚠ No profitable models found. Consider:"))
            self.stdout.write("  1. Train with more data (--years 10)")
            self.stdout.write("  2. Add more symbols to your symbols file")
            self.stdout.write("  3. Try different RR ratios (--rr-ratios '1.5,2.0,2.5')")
            self.stdout.write("  4. Adjust strategy parameters in TrendConfig")

        # Export to CSV
        if export_csv:
            export_df = df_filtered.copy()
            export_df = export_df.drop(['Train_Expected_Num', 'Val_Expected_Num'], axis=1, errors='ignore')
            export_df.to_csv(export_csv, index=False)
            self.stdout.write(f"\n✓ Results exported to: {export_csv}")

    def _find_validation_results(self, model):
        """Find validation results file for a model"""
        try:
            # Look for test results files
            results_dir = '.'
            pattern = f"test_results_{model.name}_*.json"

            import glob
            files = glob.glob(os.path.join(results_dir, pattern))

            if files:
                # Get the most recent file
                latest_file = max(files, key=os.path.getctime)
                with open(latest_file, 'r') as f:
                    data = json.load(f)

                return {
                    'precision': data.get('best_precision', 0),
                    'expected_value': data.get('expected_value', 0),
                    'best_threshold': data.get('best_threshold', 0),
                    'total_signals': data.get('test_samples', 0)
                }
        except Exception as e:
            pass
        return None

    def _analyze_best_model(self, model, show_features, show_thresholds):
        """Analyze the best model in detail"""

        self.stdout.write("\n📋 DETAILED MODEL ANALYSIS")
        self.stdout.write("-"*120)

        metrics = model.performance_metrics
        rr = model.target_rr_ratio

        self.stdout.write(f"\nModel: {model.name}")
        self.stdout.write(f"Training Date: {model.training_date}")
        self.stdout.write(f"Training Samples: {model.training_samples}")
        self.stdout.write(f"RR Ratio: {rr}:1")

        # Training metrics
        self.stdout.write("\n📊 Training Performance:")
        self.stdout.write(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")
        self.stdout.write(f"  Precision: {metrics.get('precision', 0):.2%}")
        self.stdout.write(f"  Recall: {metrics.get('recall', 0):.2%}")
        self.stdout.write(f"  F1 Score: {metrics.get('f1_score', 0):.2%}")
        self.stdout.write(f"  ROC-AUC: {metrics.get('roc_auc', 0):.2%}")

        # Confusion matrix
        tp = metrics.get('true_positives', 0)
        fp = metrics.get('false_positives', 0)
        tn = metrics.get('true_negatives', 0)
        fn = metrics.get('false_negatives', 0)

        self.stdout.write(f"\n📈 Confusion Matrix:")
        self.stdout.write(f"  True Positives:  {tp}")
        self.stdout.write(f"  False Positives: {fp}")
        self.stdout.write(f"  True Negatives:  {tn}")
        self.stdout.write(f"  False Negatives: {fn}")

        # Expected value
        precision = metrics.get('precision', 0)
        expected = (precision * rr) - ((1 - precision) * 1)
        self.stdout.write(f"\n💰 Expected Value per Trade: {expected:.3f}R")
        self.stdout.write(f"   For every $100 risked, expect ${expected*100:.0f} profit")

        # Break-even analysis
        break_even = 1 / (rr + 1)
        self.stdout.write(f"\n🎯 Break-even Analysis:")
        self.stdout.write(f"  Needed Win Rate: {break_even:.1%}")
        self.stdout.write(f"  Actual Precision: {precision:.1%}")

        if precision > break_even:
            margin = (precision - break_even) / break_even
            self.stdout.write(self.style.SUCCESS(f"  ✓ Margin: {margin:.1%} above break-even"))
        else:
            margin = (break_even - precision) / break_even
            self.stdout.write(self.style.ERROR(f"  ✗ Shortfall: {margin:.1%} below break-even"))

        # Feature importance
        if show_features:
            importance = model.feature_importance
            if importance:
                self.stdout.write("\n🔝 TOP 10 FEATURES")
                self.stdout.write("-"*60)
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (feature, imp) in enumerate(sorted_features, 1):
                    desc = self._get_feature_description(feature)
                    self.stdout.write(f"  {i:2d}. {feature:<25} {imp:.4f}  - {desc}")

        # Threshold analysis
        if show_thresholds:
            validation = self._find_validation_results(model)
            if validation:
                self.stdout.write("\n🎚️ THRESHOLD ANALYSIS")
                self.stdout.write("-"*60)
                self.stdout.write(f"Best Threshold: {validation.get('best_threshold', 'N/A')}")
                self.stdout.write(f"Expected Value at Best Threshold: {validation.get('expected_value', 'N/A')}")

    def _get_feature_description(self, feature_name):
        """Get description for feature names"""
        descriptions = {
            'stoch_oversold': 'Stochastic oscillator oversold condition',
            'stoch_overbought': 'Stochastic oscillator overbought condition',
            'stoch_cross_above': 'Stochastic cross above oversold level',
            'stoch_cross_below': 'Stochastic cross below overbought level',
            'stoch_k': 'Stochastic %K line',
            'stoch_d': 'Stochastic %D line',
            'regression_slope': 'Linear regression slope',
            'regression_cross_up': 'Regression slope turns positive',
            'regression_cross_down': 'Regression slope turns negative',
            'regression_signal': 'Regression trend direction',
            'vwap_deviation': 'Price deviation from VWAP',
            'vwap_deviation_normalized': 'Normalized VWAP deviation',
            'below_vwap': 'Price below VWAP',
            'above_vwap': 'Price above VWAP',
            'price_to_ema_fast': 'Price relative to fast EMA',
            'price_to_ema_slow': 'Price relative to slow EMA',
            'atr_pct': 'Average True Range percentage',
            'volume_ratio': 'Volume relative to average',
            'return_1d': '1-day return',
            'return_5d': '5-day return',
            'return_10d': '10-day return',
            'trend_strength': 'Trend strength (SMA ratio)',
            'is_uptrend': 'Uptrend indicator',
            'range_pct': 'Daily range percentage'
        }
        return descriptions.get(feature_name, 'Technical indicator')

