from django.urls import path
from . import views


app_name = "project2"
urlpatterns = [
    path('', views.index, name='index'),
    path('task1/', views.task1_view, name='task1'),
    path('task2/', views.task2_view, name='task2'),
]
