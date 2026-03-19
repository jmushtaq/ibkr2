from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

urlpatterns = [
    path('admin/', admin.site.urls),
    path('markets/', include('markets.urls')),
    path('', lambda request: redirect('markets:home'), name='root'),
]
