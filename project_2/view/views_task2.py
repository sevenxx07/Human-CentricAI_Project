from django.shortcuts import render

from django.http import HttpResponse
from django.template import loader


def index(request):
    context = {}

    return render(request, 'task2.html', context)
