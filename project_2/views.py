from django.shortcuts import render

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("project_base2.html")
    context = {}

    return render(request, 'project_base2.html', {})

def task1_view(request):
    # placeholder — replace with logic later
    return render(request, "task1.html")

def task2_view(request):
    # placeholder — replace with logic later
    return render(request, "task2.html")
