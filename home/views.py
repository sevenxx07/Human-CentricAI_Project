# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")

    students = [
        {"name": "Sara Vesela", "matriculation": "650869"},
        {"name": "Veronika Anokhina", "matriculation": "650885"},
        {"name": "Stina Hellgren", "matriculation": "651291"},
    ]
    
    projects = [
        {"name": "Project 1", "url_name": "project1:index"},
    ]

    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))
