import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-your-secret-key-here'

DEBUG = True

ALLOWED_HOSTS = []

INSTALLED_APPS = [
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'django.contrib.humanize',

        # Third party apps
        'django_filters',
        'django_tables2',
        'crispy_forms',
        'crispy_bootstrap5',
        'django_extensions',

        # Local apps
        'markets',
        'ml_framework',
        'ml_trend',

]

MIDDLEWARE = [
        'django.middleware.security.SecurityMiddleware',
        'django.contrib.sessions.middleware.SessionMiddleware',
        'django.middleware.common.CommonMiddleware',
        'django.middleware.csrf.CsrfViewMiddleware',
        'django.contrib.auth.middleware.AuthenticationMiddleware',
        'django.contrib.messages.middleware.MessageMiddleware',
        'django.middleware.clickjacking.XFrameOptionsMiddleware',

]

ROOT_URLCONF = 'ibkr_project.urls'

TEMPLATES = [
    {
                'BACKEND': 'django.template.backends.django.DjangoTemplates',
                'DIRS': [],
                'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                                'django.template.context_processors.debug',
                                'django.template.context_processors.request',
                                'django.contrib.auth.context_processors.auth',
                                'django.contrib.messages.context_processors.messages',
                                'markets.context_processors.dashboard_settings',

            ],

        },

    },

]

WSGI_APPLICATION = 'ibkr_project.wsgi.application'

DATABASES = {
    'default': {
                'ENGINE': 'django.db.backends.postgresql',
                'NAME': 'ibkr2_dev',
                'USER': 'dev',
                'PASSWORD': 'dev123',
                'HOST': 'localhost',
                'PORT': '5432',
#        'OPTIONS': {
#                        'options': '-c search_path=public'
#        }

    }

}

# Cache configuration
CACHES = {
    'default': {
                'BACKEND': 'django_redis.cache.RedisCache',
                'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
                        'CLIENT_CLASS': 'django_redis.client.DefaultClient',
                        #'PARSER_CLASS': 'redis.connection.HiredisParser',
                        'CONNECTION_POOL_CLASS': 'redis.BlockingConnectionPool',
            'CONNECTION_POOL_CLASS_KWARGS': {
                                'max_connections': 50,
                                'timeout': 20,

            },
                        'MAX_CONNECTIONS': 1000,
                        'PICKLE_VERSION': -1,

        },
                'KEY_PREFIX': 'ohlcv'

    }

}

CACHE_TTL = 60 * 15  # 15 minutes

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"

# Data directory
#DATA_DIR = BASE_DIR / 'data'
DATA_DIR = '/home/ubuntu/projects/ibkr/data'

MARKET_DASHBOARD_DEFAULT_COLUMNS = [
    'ticker',
    'name',
    'market_cap',
    'sector',
    'industry',
    'current_price',
    'change_1d',
    'change_1w',
    'change_1m',
    'change_1y',
]
