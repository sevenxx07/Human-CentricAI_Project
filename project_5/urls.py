from django.urls import path

from project_5 import views

app_name = "project5"
urlpatterns = [
    path('', views.index, name='index'),
]
