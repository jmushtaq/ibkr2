from django.urls import path
from . import views

app_name = 'markets'

urlpatterns = [
    path('', views.SymbolMetricsListView.as_view(), name='market_list'),
    path('home/', views.SymbolMetricsListView.as_view(), name='home'),
    path('export/', views.ExportDataView.as_view(), name='export'),
    path('chart/', views.ChartPageView.as_view(), name='chart_page'),

    path('api/chart-data/', views.ChartDataView.as_view(), name='chart_data'),
    path('api/symbols/', views.SymbolListView.as_view(), name='symbol_list'),
    path('api/indicators/', views.IndicatorDataView.as_view(), name='indicator_data'),
    path('api/indicators/list/', views.AvailableIndicatorsView.as_view(), name='indicator_list'),
]
