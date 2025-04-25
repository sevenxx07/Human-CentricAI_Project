from django.urls import path
from . import views


app_name = "project1"
urlpatterns = [
    path('', views.index, name='index'),
    path('upload_csv/', views.upload_csv, name='upload_csv'),
    path('generate_plot/', views.generate_plot, name='generate_plot'),
]
