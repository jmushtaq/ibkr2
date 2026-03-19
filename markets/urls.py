from django.urls import path
from . import views

app_name = 'markets'

urlpatterns = [
    path('', views.SymbolMetricsListView.as_view(), name='market_list'),
    path('home/', views.SymbolMetricsListView.as_view(), name='home'),
    path('api/chart-data/', views.ChartDataView.as_view(), name='chart_data'),
]
