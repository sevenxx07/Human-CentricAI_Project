from django.shortcuts import render
from django.template import loader


def index(request):
    """
    Main landing page for Project 4
    """

    template = loader.get_template("project4_base.html")

    context = {}

    return render(request, 'project4_base.html', context)


def task1_view(request):
    """
    Active Learning
    """

    context = {}

    return render(request, 'project4_coldstart.html', context)


def task2_view(request):
    """
    User Study Design & Interface
    """

    context = {}

    return render(request, 'project4_userstudy.html', context)
