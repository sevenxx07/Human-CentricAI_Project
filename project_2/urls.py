from django.urls import path
from . import view
from .view import views, views_task1, views_task2

app_name = "project2"
urlpatterns = [
    path('', views.index, name='index'),
    path('task1/', views_task1.index, name='task1'),
    path('task2/', views_task2.index, name='task2'),
]
