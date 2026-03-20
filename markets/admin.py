from django.db import models
from django.contrib import admin
from django.conf import settings
from .models import Symbol, Sector, Industry, OHLCVData, PrecomputedMetrics

class DashboardConfigAdmin(admin.ModelAdmin):
    """Admin interface for dashboard configuration"""

    list_display = ['name', 'value']

    def get_queryset(self, request):
        # This is a simple way to expose settings in admin
        # You might want to create a proper model for this
        return super().get_queryset(request)

# You can create a simple model for dashboard config if needed
# models.py
class DashboardConfig(models.Model):
    """Store dashboard configuration"""
    name = models.CharField(max_length=100, unique=True)
    value = models.JSONField()
    description = models.TextField(blank=True)

    class Meta:
        verbose_name = "Dashboard Configuration"
        verbose_name_plural = "Dashboard Configurations"

    def __str__(self):
        return self.name

# Then register it in admin.py
admin.site.register(DashboardConfig, DashboardConfigAdmin)

