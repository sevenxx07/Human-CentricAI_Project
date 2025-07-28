from django.urls import path

from project_4.view import views_task2, views_task1, views

app_name = "project4"
urlpatterns = [
    path('', views.index, name='index'),
    path('task1/', views_task1.index, name='task1'),
    path('task2/', views_task2.index, name='task2'),
]

