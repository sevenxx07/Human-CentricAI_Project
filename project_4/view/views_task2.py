from django.shortcuts import render
from django.template import loader

def index(request):
    """
    """

    template = loader.get_template("project4_userstudy.html")

    context = {}

    return render(request, 'project4_userstudy.html', context)
