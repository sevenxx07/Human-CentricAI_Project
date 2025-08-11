from django.urls import path
from . import views


app_name = "project5"
urlpatterns = [
    path('', views.index, name='index'),
]
