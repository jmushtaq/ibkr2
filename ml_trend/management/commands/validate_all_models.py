#!/usr/bin/env python
import os
import sys
import django
from django.core.management import call_command

sys.path.append('/home/ubuntu/projects/ibkr2')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ibkr_project.settings')
django.setup()

from ml_trend.models import MLTrendModel

def validate_all_models():
    """Validate all active models"""

    # Get all active models
    models = MLTrendModel.objects.filter(is_active=True).order_by('-created_at')

    print("\n" + "="*80)
    print("VALIDATING ALL MODELS")
    print("="*80)

    results = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"Validating: {model.name}")
        print(f"  RR Ratio: {model.target_rr_ratio}")
        print(f"  Training Accuracy: {model.performance_metrics.get('accuracy', 0):.2%}")
        print(f"{'='*60}")

        try:
            # Run validation
            call_command('validate_model',
                        model_id=model.id,
                        test_years=2)
            results.append({
                'model': model.name,
                'success': True
            })
        except Exception as e:
            print(f"Error validating {model.name}: {e}")
            results.append({
                'model': model.name,
                'success': False,
                'error': str(e)
            })

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"{status} {r['model']}")

if __name__ == "__main__":
    validate_all_models()
