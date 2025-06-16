import base64
import numpy as np
import pandas as pd
import os
import ast

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.http import JsonResponse
from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
from sklearn.metrics import accuracy_score 

from project_3.ML_models.LogisticRegressionmodel import LogisticRegressionModel
from project_3.ML_models.DT import DT

DEBUG = True  # Set to False in production
model_global = None  # Global variable to hold the model instance
vectorizer_global = None  # Global variable to hold the vectorizer instance


def index(request):
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'decision_tree_visualization':
            decision_tree_visualization(request, context)
        elif action == 'log_regression_visualization':
            logistic_regression_visualization(request, context)

    return render(request, 'project3_base.html', context)


# NOT FINISHED
def decision_tree_visualization(request, context):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(PROJECT_ROOT, 'data', 'project3_data')

    selected_lambda = float(request.POST.get('lambda', 0.1))
    print('Selected model:', selected_lambda)

    DT_results = DT().fit_and_return_results(lambda_val = selected_lambda)

    return render(request, 'task1.html', context)


def logistic_regression_visualization(request, context):

    return render(request, 'task2.html', context)


