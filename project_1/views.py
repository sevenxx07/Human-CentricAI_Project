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
        elif action == "train" and 'csv_loaded' in request.POST:
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
            context['scroll_to'] = request.POST.get('scroll_to')
            DATA_STORAGE['csv_name'] = csv_file.name
    except Exception as e:
        context['error'] = f"Error reading CSV: {str(e)}"


def handle_data_split(size):
    """
    Handles the data split into features and labels.
    """

    df = DATA_STORAGE.get('df')
    # Split dataset: last column is label, rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # Ask the user how much
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size / 100, random_state=42)

    DATA_STORAGE['X_train'] = X_train
    DATA_STORAGE['X_test'] = X_test
    DATA_STORAGE['y_train'] = y_train
    DATA_STORAGE['y_test'] = y_test


def handle_train_model(request, context):
    context['selected_model'] = DATA_STORAGE.get('model_type')
    context['csv_uploaded'] = True
    context['uploaded_filename'] = DATA_STORAGE.get('csv_name')
    context['split_success'] = True
    context['scroll_to'] = request.POST.get('scroll_to')
    context['selected_model'] = DATA_STORAGE.get('model_type')
    context['selected_C'] = DATA_STORAGE.get('selected_C')
    context['selected_max_depth'] = DATA_STORAGE.get('selected_max_depth')
    context['selected_k'] = DATA_STORAGE.get('selected_k')
    context['test_size'] = DATA_STORAGE.get('test_size')

    if 'scatter_url' in DATA_STORAGE:
        context['scatter_url'] = DATA_STORAGE['scatter_url']
        context['image_url'] = DATA_STORAGE['scatter_url']
        context['selected_feature1'] = DATA_STORAGE.get('selected_feature1')
        context['selected_feature2'] = DATA_STORAGE.get('selected_feature2')

    X_train = DATA_STORAGE.get('X_train')
    y_train = DATA_STORAGE.get('y_train')

    if X_train is None or y_train is None:
        context['error'] = "No training data available."
        return

    # Loading model from previous choice
    model = DATA_STORAGE.get('model')
    model.X = X_train
    model.Y = y_train

    print("Training model with data:", X_train.head(), y_train.head())
    print("Model type selected:", DATA_STORAGE['model_type'])
    print("Model parameters:", model.__dict__)

    # Train the model
    model.train()

    print("Trained parameters:", model.__dict__)

    DATA_STORAGE['model'] = model
    context['train_success'] = True

    #Predict and evaluate
    x_test = DATA_STORAGE.get('X_test')
    y_test = DATA_STORAGE.get('y_test')
    predictions = model.predict(x_test)
    metrics = model.evaluate(y_test, predictions)
    context.update({
        'accuracy': round(metrics['accuracy'] * 100, 2),
        'precision': round(metrics['precision'] * 100, 2),
        'recall': round(metrics['recall'] * 100, 2),
        'f1': round(metrics['f1'] * 100, 2),
    })
    df = DATA_STORAGE.get('df')
    print("Test set size:", len(y_test))
    print("Predictions:", predictions[:10])
    print("Ground truth:", y_test[:10].tolist())

    feature1 = DATA_STORAGE.get('selected_feature1') or df.columns[0]
    feature2 = DATA_STORAGE.get('selected_feature2') or df.columns[1]

    output_url, _, _ = generate_plot(df, model=model, feature1=feature1, feature2=feature2, filename_prefix="output")
    context['output_plot_url'] = output_url
    DATA_STORAGE['output_plot_url'] = output_url


def handle_model_selection(request, context):
    """
    Handles the model selection from the dropdown menu.
    """
    model_type = request.POST.get('model')
    DATA_STORAGE['model_type'] = model_type

    context['selected_model'] = model_type
    context['csv_uploaded'] = True
    context['uploaded_filename'] = DATA_STORAGE.get('csv_name')
    context['scroll_to'] = request.POST.get('scroll_to')
    if 'scatter_url' in DATA_STORAGE:
        context['scatter_url'] = DATA_STORAGE['scatter_url']
        context['image_url'] = DATA_STORAGE['scatter_url']
        context['selected_feature1'] = DATA_STORAGE.get('selected_feature1')
        context['selected_feature2'] = DATA_STORAGE.get('selected_feature2')

    if model_type == 'logistic':
        C = float(request.POST.get('C', 1.0))
        context['selected_C'] = str(C)
        DATA_STORAGE['selected_C'] = str(C)
        model = LogisticReg(None, None, C, 1000)
        print("Model type selected:", model_type, C)
    elif model_type == 'tree':
        max_depth = int(request.POST.get('max_depth', 5))
        context['selected_max_depth'] = str(max_depth)
        DATA_STORAGE['selected_max_depth'] = str(max_depth)
        model = DecisionTree(None, None, max_depth)
        print("Model type selected:", model_type, max_depth)
    elif model_type == 'knn':
        k = int(request.POST.get('k', 5))
        context['selected_k'] = str(k)
        DATA_STORAGE['selected_k'] = str(k)
        model = KNN(None, None, k)
        print("Model type selected:", model_type, k)
    else:
        context['error'] = "Unsupported model selected."
        return

    DATA_STORAGE['model'] = model
    test_size = float(request.POST.get('test_size', 0.2))
    context['test_size'] = str(int(test_size))
    DATA_STORAGE['test_size'] = int(test_size)
    handle_data_split(test_size)


def generate_plot(df, model=None, feature1=None, feature2=None, filename_prefix="plot"):
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt

    if feature1 is None or feature2 is None:
        # Default to first two columns if not provided
        feature1, feature2 = df.columns[:2]

    X_full = df.iloc[:, :-1]  # all features for prediction
    X_vis = df[[feature1, feature2]]  # just these 2 for plotting
    y = df.iloc[:, -1]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    plt.figure(figsize=(6, 4))

    if model:
        try:
            y_pred = model.predict(X_full)
            y_pred_encoded = le.fit_transform(y_pred)
            scatter = plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y_pred_encoded, cmap='coolwarm', alpha=0.7, marker='x')
            plt.title("Model Predictions")
            labels = le.classes_
            for i, label in enumerate(labels):
                plt.scatter([], [], c=plt.cm.coolwarm(i / len(labels)), marker='x', label=label)
        except Exception as e:
            print(f"[plot error] {e}")
            return None, feature1, feature2
    else:
        scatter = plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y_encoded, cmap='viridis', alpha=0.7)
        plt.title("Raw Data")
        labels = le.classes_
        for i, label in enumerate(labels):
            plt.scatter([], [], c=plt.cm.viridis(i / len(labels)), label=label)

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend(loc='best')

    filename = f"{filename_prefix}_{uuid.uuid4().hex}.png"
    path = os.path.join(settings.MEDIA_ROOT, filename)
    plt.savefig(path)
    plt.close()

    return settings.MEDIA_URL + filename, feature1, feature2

def handle_plot_generation(request, context):
    df = DATA_STORAGE.get('df')
    if df is None:
        context['error'] = "No CSV uploaded."
        return

    context['csv_uploaded'] = True
    context['uploaded_filename'] = DATA_STORAGE.get('csv_name')
    context['scroll_to'] = request.POST.get('scroll_to')

    # Let the user still pick features in Step 1.5 — optional
    feature1 = request.POST.get('feature1') or df.columns[0]
    feature2 = request.POST.get('feature2') or df.columns[1]

    plot_url, f1, f2 = generate_plot(df, None, feature1, feature2, filename_prefix="raw_plot")
    context['scatter_url'] = plot_url
    context['image_url'] = plot_url
    context['selected_feature1'] = f1
    context['selected_feature2'] = f2

    # Store for later reuse
    DATA_STORAGE['scatter_url'] = plot_url
    DATA_STORAGE['selected_feature1'] = f1
    DATA_STORAGE['selected_feature2'] = f2


    #         # Plot the predicted labels (from the model)
    #         scatter_pred = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred_encoded, cmap='coolwarm', alpha=0.7, marker='x', label="Predicted Labels")
    #     except Exception as e:
    #         context['error'] = f"Model prediction failed: {str(e)}"
    #         return
        
    # plt.colorbar(scatter_true, label="True Class")
    # plt.xlabel(feature1)
    # plt.ylabel(feature2)

    # plt.legend(loc='best')

    # filename = f"plot_{uuid.uuid4().hex}.png"
    # image_path = os.path.join(settings.MEDIA_ROOT, filename)
    # plt.savefig(image_path)
    # plt.close()

    # # Pass the plot URL back to the context
    # context['scatter_url'] = settings.MEDIA_URL + filename
    # context['selected_feature1'] = feature1
    # context['selected_feature2'] = feature2
 ##
    # plt.title(f"Visualization of {feature1} vs {feature2}")
    # plt.colorbar(scatter, label="True Class")
    # plt.xlabel(feature1)
    # plt.ylabel(feature2)

    # # Save the plot
    # filename = f"plot_{uuid.uuid4().hex}.png"
    # image_path = os.path.join(settings.MEDIA_ROOT, filename)
    # plt.savefig(image_path)
    # plt.close()

    # # Pass the plot URL back to the context
    # context['scatter_url'] = settings.MEDIA_URL + filename
    # context['selected_feature1'] = feature1
    # context['selected_feature2'] = feature2

##

    # # Loading model from previous choice
    # model_type = request.POST.get('model')
    # context['selected_model'] = model_type
    # context['csv_uploaded'] = True
    # context['uploaded_filename'] = DATA_STORAGE.get('csv_name')


    # model = DATA_STORAGE.get("model")
    # X = df.iloc[:, :-1]
    # y = df.iloc[:, -1]

    # le = LabelEncoder()
    # y_encoded = le.fit_transform(y)
    # #print("Encoded labels:", y_encoded[:10])

    # plt.figure(figsize=(6, 4))

    # if model and "X_test" in DATA_STORAGE and "y_test" in DATA_STORAGE: 
    #     X_test = DATA_STORAGE['X_test']
    #     y_test = DATA_STORAGE['y_test']

    #     y_test_encoded = le.transform(y_test)
    #     try: 
    #         y_pred = model.predict(X_test)
    #     except Exception as e: 
    #         context['error'] = f" Model prediction failed: {str(e)} "
    #         return 
        
    #     scatter = plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='coolwarm', alpha=0.7)
    #     plt.title("Predicted Labels (first two features)")
    #     plt.colorbar(scatter, label="Predicted Class")
    #     plt.xlabel(X_test.columns[0])
    #     plt.ylabel(X_test.columns[1])
    # else:
    #     scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_encoded, cmap='viridis', alpha=0.7)
    #     plt.title("Raw Data (first two features)")
    #     plt.colorbar(scatter, label="True Class")
    #     plt.xlabel(X.columns[0])
    #     plt.ylabel(X.columns[1])

    # filename = f"plot_{uuid.uuid4().hex}.png"
    # image_path = os.path.join(settings.MEDIA_ROOT, filename)
    # plt.savefig(image_path)
    # plt.close()

    # context['image_url'] = settings.MEDIA_URL + filename
    
    
