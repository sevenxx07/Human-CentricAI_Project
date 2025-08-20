from django.urls import path

from project_4.view import unguided, guided, views

app_name = "project4"
urlpatterns = [
    path('', views.index, name='index'),
    path('guided/', guided.index, name='guided'),
    path('unguided/', unguided.index, name='unguided'),
]

