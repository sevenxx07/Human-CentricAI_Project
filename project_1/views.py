import csv
import io
import os
import uuid

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from django.conf import settings
from django.shortcuts import render
from .forms import CSVUploadForm
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import default_storage
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_STORAGE = {}

def index(request):
    context = {}
    global DATA_STORAGE

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'upload' and 'csv_file' in request.FILES:
            # Read and store the uploaded CSV
            csv_file = request.FILES['csv_file']
            try:
                df = pd.read_csv(csv_file)

                if df.shape[1] < 3:
                    context['error'] = "Dataset must contain at least two features and one label column."
                else:
                    DATA_STORAGE['df'] = df
                    context['column_names'] = list(df.columns)
                    context['data_preview'] = df.head(10).values.tolist()
                    context['csv_uploaded'] = True
                    context['uploaded_filename'] = csv_file.name
            except Exception as e:
                context['error'] = f"Error reading CSV: {str(e)}"

        elif action == 'plot' and 'csv_loaded' in request.POST:
            df = DATA_STORAGE.get('df')
            if df is None:
                context['error'] = "No CSV uploaded."
            else:
                model_type = request.POST.get('model')

                # Feature columns and label
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                # Encode target if classification
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                if model_type == 'logistic':
                    C = float(request.POST.get('C', 1.0))
                    model = LogisticRegression(C=C, max_iter=1000)
                elif model_type == 'tree':
                    max_depth = int(request.POST.get('max_depth', 5))
                    model = DecisionTreeClassifier(max_depth=max_depth)
                else:
                    context['error'] = "Unsupported model selected."
                    return render(request, 'project1/index.html', context)

                model.fit(X_train, y_train)

                # Plot first two features
                plt.figure(figsize=(6, 4))
                scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='viridis')
                plt.xlabel(X.columns[0])
                plt.ylabel(X.columns[1])
                plt.title(f"{model.__class__.__name__} Result")
                plt.colorbar(scatter)

                filename = f"plot_{uuid.uuid4().hex}.png"
                image_path = os.path.join(settings.MEDIA_ROOT, filename)
                plt.savefig(image_path)
                plt.close()

                context['image_url'] = settings.MEDIA_URL + filename
                context['column_names'] = list(df.columns)
                context['data_preview'] = df.head(10).values

    return render(request, 'project_base.html', context)

