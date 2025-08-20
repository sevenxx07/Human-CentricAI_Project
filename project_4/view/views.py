from django.shortcuts import render
from django.template import loader


def index(request):
    """
    Landing page for Project 4 - Cold Start Movie Recommender Study
    """
    return render(request, 'project4_base.html')


def guided(request):
    """
    Main cold start recommendation interface
    """
    # Get the mode from URL parameter
    mode = request.GET.get('mode', 'guided')  # default to guided

    # Store mode in session for later use
    request.session['study_mode'] = mode

    context = {
        'study_mode': mode,
    }

    return render(request, 'project4_coldstart.html', context)