import matplotlib
from sklearn.model_selection import train_test_split

from .ML_models.decision_tree import DecisionTree
from .ML_models.knn import KNN
from .ML_models.logistic_reg import LogisticReg

matplotlib.use('Agg')  # Force non-GUI backend

# Standard libraries
import os
import uuid

# Data science libraries
from matplotlib import pyplot as plt
import pandas as pd

# Django utilities
from django.conf import settings
from django.shortcuts import render

# Import our own ML model classes

# Django form for file upload
#from .forms import ModelSelectionForm
from sklearn.preprocessing import LabelEncoder

# In-memory storage of uploaded data (temporary)
DATA_STORAGE = {}


def index(request):
    """
    The main view that handles the CSV upload, model selection, and plotting.
    """

    context = {}  # This will be passed into the THMl template
    global DATA_STORAGE  # Making sure we can

    if request.method == 'POST':
        action = request.POST.get('action')  # Get the action from the form (upload, select, plot)
        print("ACTION:", action)

        if action == 'upload' and 'csv_file' in request.FILES:
            handle_csv_upload(request, context)
        elif action == 'select' and 'csv_loaded' in request.POST:
            handle_model_selection(request, context)
        elif action == 'plot' and 'csv_loaded' in request.POST:
            handle_plot_generation(request, context)
        elif action == "split" and 'csv_loaded' in request.POST:
            handle_data_split(request, context)
        elif action == "train" and 'csv_loaded' and 'model_type' in request.POST:
            handle_train_model(request, context)

    return render(request, 'project_base.html', context)


def handle_csv_upload(request, context):
    """
    Handles the CSV file upload, reads it into a DataFrame, and stores it in memory.
    """

    csv_file = request.FILES['csv_file']
    try:
        df = pd.read_csv(csv_file)

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


def handle_data_split(request, context):
    """
    Handles the data split into features and labels.
    """
    context['selected_model'] = DATA_STORAGE.get('model_type')
    context['uploaded_filename'] = DATA_STORAGE.get('csv_name')
    context['csv_uploaded'] = True

    df = DATA_STORAGE.get('df')
    if df is None:
        context['error'] = "No CSV uploaded."
        return

    # Split dataset: last column is label, rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Ask the user how much
    test_size = float(request.POST.get('test_size', 0.2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

    DATA_STORAGE['X_train'] = X_train
    DATA_STORAGE['X_test'] = X_test
    DATA_STORAGE['y_train'] = y_train
    DATA_STORAGE['y_test'] = y_test

    context['split_success'] = True


def handle_train_model(request, context):
    context['selected_model'] = DATA_STORAGE['model_type']
    context['csv_uploaded'] = True
    context['uploaded_filename'] = DATA_STORAGE.get('csv_name')

    X_train = DATA_STORAGE.get('X_train')
    y_train = DATA_STORAGE.get('y_train')

    if X_train is None or y_train is None:
        context['error'] = "No training data available."
        return

    # Loading model from previous choice
    model = DATA_STORAGE.get('model')
    model.X = X_train
    model.Y = y_train

    # Train the model
    model.train()

    DATA_STORAGE['model'] = model
    context['train_success'] = True


def handle_model_selection(request, context):
    """
    Handles the model selection from the dropdown menu.
    """
    model_type = request.POST.get('model')
    DATA_STORAGE['model_type'] = model_type

    context['selected_model'] = model_type
    context['csv_uploaded'] = True
    context['uploaded_filename'] = DATA_STORAGE.get('csv_name')

    print("Model type selected:", model_type)

    if model_type == 'logistic':
        C = float(request.POST.get('C', 1.0))
        context['selected_C'] = request.POST.get('C', '1.0')
        model = LogisticReg(None, None, C, 1000)
    elif model_type == 'tree':
        max_depth = int(request.POST.get('max_depth', 5))
        context['selected_max_depth'] = request.POST.get('max_depth', '5')
        model = DecisionTree(None, None, max_depth)
    elif model_type == 'knn':
        k = int(request.POST.get('k', 5))
        context['selected_k'] = request.POST.get('k', '5')
        model = KNN(None, None, k)
    else:
        context['error'] = "Unsupported model selected."
        return

    DATA_STORAGE['model'] = model


def handle_plot_generation(request, context):
    """
    Handles the plot generation based on the selected model and the uploaded CSV.
    """
    df = DATA_STORAGE.get('df')
    if df is None:
        context['error'] = "No CSV uploaded."
        return

    # Loading model from previous choice
    model_type = request.POST.get('model')
    context['selected_model'] = model_type
    context['csv_uploaded'] = True
    context['uploaded_filename'] = DATA_STORAGE.get('csv_name')


    model = DATA_STORAGE.get("model")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    #print("Encoded labels:", y_encoded[:10])

    plt.figure(figsize=(6, 4))

    if model and "X_test" in DATA_STORAGE and "y_test" in DATA_STORAGE: 
        X_test = DATA_STORAGE['X_test']
        y_test = DATA_STORAGE['y_test']

        y_test_encoded = le.transform(y_test)
        try: 
            y_pred = model.predict(X_test)
        except Exception as e: 
            context['error'] = f" Model prediction failed: {str(e)} "
            return 
        
        scatter = plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='coolwarm', alpha=0.7)
        plt.title("Predicted Labels (first two features)")
        plt.colorbar(scatter, label="Predicted Class")
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])
    else:
        scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_encoded, cmap='viridis', alpha=0.7)
        plt.title("Raw Data (first two features)")
        plt.colorbar(scatter, label="True Class")
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])

    filename = f"plot_{uuid.uuid4().hex}.png"
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    plt.savefig(image_path)
    plt.close()

    context['image_url'] = settings.MEDIA_URL + filename
    
    
    # # Plotting just the first two features
    
    # # scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='viridis')
    # scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_encoded, cmap='viridis')
    # plt.xlabel(X.columns[0])
    # plt.ylabel(X.columns[1])
    # plt.colorbar(scatter)

    # # Save the plot image to the media
    # filename = f"plot_{uuid.uuid4().hex}.png"
    # image_path = os.path.join(settings.MEDIA_ROOT, filename)
    # plt.savefig(image_path)
    # plt.close()

    # # Pass the image URL to the HTML template
    # context['image_url'] = settings.MEDIA_URL + filename
    # context['column_names'] = list(df.columns)
    # context['data_preview'] = df.head(10).values
