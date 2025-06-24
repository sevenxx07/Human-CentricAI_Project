from django.urls import path
from . import views

app_name = "project3"
urlpatterns = [
    path('', views.index, name='index'),
    path('get_samples/', views.get_samples, name="get_samples"),
]