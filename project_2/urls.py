from django.urls import path
from . import views


app_name = "project2"
urlpatterns = [
    path('', views.index, name='index'),
]
