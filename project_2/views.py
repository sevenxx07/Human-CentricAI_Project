from django.shortcuts import render

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("project_base2.html")
    context = {}

    return render(request, 'project_base2.html', {})
