from django.urls import path
from . import views

urlpatterns = [
        path('api/predictions/<str:symbol_ticker>/', views.get_predictions, name='get_predictions'),
        path('api/models/<int:model_id>/', views.get_model_performance, name='get_model_performance'),
        path('api/models/', views.list_models, name='list_models'),
        path('api/predict/', views.generate_prediction, name='generate_prediction'),
        path('api/predict/simple/', views.generate_simple_prediction, name='generate_simple_prediction'),
        path('api/backtests/<int:backtest_id>/', views.get_backtest_results, name='get_backtest_results'),
        path('dashboard/', views.dashboard, name='dashboard'),
]
