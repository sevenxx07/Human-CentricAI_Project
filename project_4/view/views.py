from django.shortcuts import render, redirect


def index(request):
    """
    Landing page for Project 4 - Cold Start Movie Recommender Study
    Handles consent form and study selection
    """
    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'consent':
            # Handle consent form submission
            consent_participate = request.POST.get('consent_participate')
            consent_data = request.POST.get('consent_data')
            consent_age = request.POST.get('consent_age')
            consent_publication = request.POST.get('consent_publication')

            # Check required consents
            if consent_participate and consent_data and consent_age:
                # Store consent in session
                request.session['consent_given'] = True
                request.session['consent_participate'] = True
                request.session['consent_data'] = True
                request.session['consent_age'] = True
                request.session['consent_publication'] = bool(consent_publication)

                # Redirect to same page to show study selection
                return redirect('project4:index')
            else:
                # Return with error if required consents not given
                context = {
                    'consent_given': False,
                    'consent_errors': True
                }
                return render(request, 'project4_base.html', context)

        elif action == 'study_mode':
            # Handle study mode selection (only if consent was given)
            if request.session.get('consent_given'):
                study_mode = request.POST.get('study_mode')
                request.session['study_mode'] = study_mode

                # Clear ALL consent data when leaving the base page to go to study
                request.session.pop('consent_given', None)
                request.session.pop('consent_participate', None)
                request.session.pop('consent_data', None)
                request.session.pop('consent_age', None)
                request.session.pop('consent_publication', None)

                # Redirect to the appropriate view based on study mode
                if study_mode == 'guided':
                    return redirect('project4:guided')
                elif study_mode == 'unguided':
                    return redirect('project4:unguided')
                else:
                    return redirect('project4:guided')  # default fallback
            else:
                # Redirect back if no consent given
                return redirect('project4:index')

    # GET request - check if we have consent or need to clear it
    consent_given = request.session.get('consent_given', False)

    # If this is a fresh visit (no referrer from our own site), clear consent
    # This ensures users must consent again if they return to the base page from outside
    if not consent_given:
        request.session.pop('consent_given', None)
        request.session.pop('consent_participate', None)
        request.session.pop('consent_data', None)
        request.session.pop('consent_age', None)
        request.session.pop('consent_publication', None)

    context = {
        'consent_given': consent_given,
        'consent_errors': False
    }

    return render(request, 'project4_base.html', context)


def guided(request):
    """
    Main cold start recommendation interface for guided mode
    """
    # Check if user came from proper consent flow
    study_mode = request.session.get('study_mode')
    if not study_mode:
        # Redirect back to consent if no proper session
        return redirect('project4:index')

    context = {
        'study_mode': 'guided',
        'session_active': True,
    }

    return render(request, 'project4_guided.html', context)


def unguided(request):
    """
    Main cold start recommendation interface for unguided mode
    """
    # Check if user came from proper consent flow
    study_mode = request.session.get('study_mode')
    if not study_mode:
        # Redirect back to consent if no proper session
        return redirect('project4:index')

    context = {
        'study_mode': 'unguided',
        'session_active': True,
    }

    return render(request, 'project4_unguided.html', context)