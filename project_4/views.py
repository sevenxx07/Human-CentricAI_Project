from django.shortcuts import render

def index(request):
    context = {}
    
    if request.method == 'POST':
        action = request.POST.get('action')


    return render(request, 'project4_base.html', context)

