import base64
import numpy as np
import pandas as pd
import os
import ast
import graphviz
from gosdt import GOSDTClassifier 

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.http import JsonResponse, FileResponse
from django.conf import settings
from sklearn.base import clone

from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
from sklearn.metrics import accuracy_score 
import seaborn as sns
from palmerpenguins import load_penguins
import pandas as pd 

from project_3.Interpretability.Decision_tree_complexity import SparseDecisionTree
from project_3.Interpretability.Decision_tree import PalmerPenguinsDecisionTree

from project_3.Counterfactuals.counterfactuals_workflow import CounterfactualExplainer


def index(request):
    context = {}

    df = load_penguins().dropna()
    context['dataset'] = df
    dataset_sample = df.sample(n=5, random_state = np.random.randint(1,10000))
    context["dataset_head"] = dataset_sample.to_dict(orient="records")

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'normal_DT':
            DT(request, context)
        
        elif action == 'sparse_DT':
            sparse_DT(request, context)

        elif action == 'log_regression':
            logistic_regression(request, context)

        elif action == 'counterfactual':
            counterfactual(request, context)

    return render(request, 'project3_base.html', context)

def DT(request, context):

    tree_model = PalmerPenguinsDecisionTree()

    tree_model.train_model()
    metrics_normal = tree_model.get_metrics()
    img_path = tree_model.generate_tree_visualization(save_path = 'static/tree.png')
    
    with open(img_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    context.update({
        "metrics_normal": True,
        "test_accuracy_normal": metrics_normal['test_accuracy'], 
        "depth_normal": metrics_normal['tree_depth'], 
        "n_nodes_normal": metrics_normal['num_leaves'],
        "total_nodes_normal": metrics_normal['total_nodes'],
        "tree_image": img_base64,
        "show_tree": True
    })

    return render(request, 'project3_base.html', context)

# NOT FINISHED 
def sparse_DT(request, context):

    selected_lambda = float(request.POST.get('lambda', 0.1))
    print('Selected model:', selected_lambda)

    penguins = sns.load_dataset('penguins').dropna().reset_index(drop=True)
    X = penguins.select_dtypes(include=['number'])
    y = penguins['species']

    model = SparseDecisionTree(alpha=selected_lambda)
    model.fit(X, y)

    # Get metrics
    metrics_sparse = model.get_metrics(X, y)

    # Generate visualization and convert to base64
    image_path = 'static/tree.png'
    model.export_tree_graphviz(X.columns, save_path=image_path)

    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    context.update({
        "metrics_sparse": True,
        'lambda': selected_lambda,
        "test_accuracy_sparse": metrics_sparse['test_acc'],
        "depth_sparse": metrics_sparse['tree_depth'],
        "n_nodes_sparse": metrics_sparse['num_leaves'],
        "total_nodes_sparse": metrics_sparse['total_nodes'],
        "tree_image_sparse": img_base64,
        'message': f"Tree trained with lambda (alpha): {selected_lambda}"
    })

def logistic_regression(request, context):
    pass
        #return render(request, 'task3.html', context)

def counterfactual(request, context):
    penguins = sns.load_dataset('penguins').dropna().reset_index(drop=True)
    print(penguins.head())
    features = penguins.select_dtypes(include=['number']).columns.tolist()
    X = penguins[features]
    y = penguins['species']

    tree_model = SparseDecisionTree(alpha = 0.1)
    tree_model.fit(X,y)

    try:
        example_index = int(request.POST.get('example_index', 0))
        target_label = request.POST.get('target_label', 'Adelie')
        N = int(request.POST.get('N', 500))
        k = int(request.POST.get('k', 3))
    except Exception as e:
        context['error'] = f"Invalid input: {e}"
        return

    if example_index < 0 or example_index >= len(penguins):
        context['error'] = "Example index out of range."
        return

    x = penguins.iloc[example_index]

    # Instantiate your counterfactual explainer
    explainer = CounterfactualExplainer(
        model=tree_model,
        data=penguins,
        N=N,
        k=k
    )

    # Compute counterfactuals
    counterfactuals = explainer.compute(x, target_label)

    context.update({
        'counterfactuals': counterfactuals,
        'selected_example': x.to_dict(),
        'target_label': target_label,
        'N': N,
        'k': k,
        'features': features,
        'species_options': penguins['species'].unique().tolist(),
        'example_index': example_index,
        'message': f'Found {len(counterfactuals)} counterfactual explanations.'
    })

    return render(request, 'project3_base.html', context)



def get_samples(request, context):
    species = request.GET.get('species', '')
    print(f"get_samples called with species={species}")
    penguins = sns.load_dataset('penguins').dropna().reset_index()
    print(penguins.head())
    if species: 
        filtered = penguins[penguins['species'] == species].copy()
    else:
        filtered = penguins.copy()

    samples = filtered.to_dict(orient='records')
    print(f"Returning {len(samples)} samples")
    return JsonResponse({'samples': samples})