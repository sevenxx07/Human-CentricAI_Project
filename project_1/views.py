# Standard libraries
import csv
import io
import os
import uuid

# Data science libraries 
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Django utilities
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import default_storage

# Import our own ML model classes 
from .ML_models.logistic_reg import LogisticReg
from .ML_models.decision_tree import DecisionTree
from .ML_models.knn import KNN


 # Django form for file upload 
from .forms import CSVUploadForm

# Scikit-learn for ML support (if needed)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# In-memory storage of uploaded data (temporary)
DATA_STORAGE = {}

# 
def index(request):
    context = {} # This will be passed into the THMl template
    global DATA_STORAGE # Making sure we can 

    if request.method == 'POST':
        action = request.POST.get('action') #Get the action from the form (upload, select, plot)

# ----------- 1. Uploading a CSV ------------
        if action == 'upload' and 'csv_file' in request.FILES:
            # Read and store the uploaded CSV
            csv_file = request.FILES['csv_file']
            try:
                df = pd.read_csv(csv_file) #Read CSV into pandas dataframe

                if df.shape[1] < 3:
                    # If less than 2 features + 1 label, show an error 
                    context['error'] = "Dataset must contain at least two features and one label column."
                else:
                    # Store dataframe in memory for later use 
                    DATA_STORAGE['df'] = df
                    context['column_names'] = list(df.columns)
                    context['data_preview'] = df.head(10).values.tolist()
                    context['csv_uploaded'] = True
                    context['uploaded_filename'] = csv_file.name
                    DATA_STORAGE['csv_name'] = csv_file.name
            except Exception as e:
                context['error'] = f"Error reading CSV: {str(e)}"

# ------------ 2. Selecting a model --------------
        elif action == 'select' and 'csv_loaded' in request.POST:
            model_type = request.POST.get('model') # e.g., logistic tree, knn
            context['selected_model'] = model_type
            context['csv_uploaded'] = True
            context['uploaded_filename'] = DATA_STORAGE.get('csv_name')

# ------------ 3. Plotting the data and training model -----------
        elif action == 'plot' and 'csv_loaded' in request.POST:
            df = DATA_STORAGE.get('df') #Load stored CSV
            if df is None:
                context['error'] = "No CSV uploaded."
            else:
                # Get model choice again
                model_type = request.POST.get('model')
                context['selected_model'] = model_type
                context['csv_uploaded'] = True
                context['uploaded_filename'] = DATA_STORAGE.get('csv_name')

                # Split dataset: last column is label, rest are reatures
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                # Instantiate the appropriate model
                if model_type == 'logistic':
                    C = float(request.POST.get('C', 1.0))
                    context['selected_C'] = request.POST.get('C', '1.0')
                    model = LogisticReg(X, y, C, 1000)
                elif model_type == 'tree':
                    max_depth = int(request.POST.get('max_depth', 5))
                    context['selected_max_depth'] = request.POST.get('max_depth', '5')
                    model = DecisionTree(X, y, max_depth)
                elif model_type == 'knn':
                    k = int(request.POST.get('k', 5))
                    context['selected_k'] = request.POST.get('k', '5')
                    model = KNN(X, y, k)
                else:
                    context['error'] = "Unsupported model selected."
                    return render(request, 'project1/index.html', context)

# ------------- 4. Plotting the data ----------------
                # Plotting just the first two features
                plt.figure(figsize=(6, 4))
                scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='viridis')
                plt.xlabel(X.columns[0])
                plt.ylabel(X.columns[1])
                plt.colorbar(scatter)

                # Save the plot image to the media
                filename = f"plot_{uuid.uuid4().hex}.png"
                image_path = os.path.join(settings.MEDIA_ROOT, filename)
                plt.savefig(image_path)
                plt.close()

                # Pass the image URL to the HTML template 
                context['image_url'] = settings.MEDIA_URL + filename
                context['column_names'] = list(df.columns)
                context['data_preview'] = df.head(10).values

    # Render the main page 
    return render(request, 'project_base.html', context)

