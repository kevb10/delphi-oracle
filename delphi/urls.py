from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('<ticker_name>', views.predict, name='predict'),
]
