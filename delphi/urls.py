from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('load/<ticker_name>', views.predict, name='predict'),
    path('evaluate/<ticker_name>', views.evaluate, name='evaluate'),
]
