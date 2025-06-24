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

from project_3.Interpretability.Decision_tree_complexity import SparseDecisionTree
from project_3.Interpretability.Decision_tree import PalmerPenguinsDecisionTree

from project_3.Counterfactuals.counterfactuals_workflow import CounterfactualExplainer


def index(request):
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'decision_tree_visualization':
            DT(request, context)

        elif action == 'log_regression_visualization':
            logistic_regression_visualization(request, context)

        elif action == 'counterfactual':
            counterfactual(request, context)

    return render(request, 'project3_base.html', context)


def DT(request, context):

    selected_lambda = float(request.POST.get('lambda', 0.1))
    print('Selected model:', selected_lambda)

    # penguins = sns.load_dataset('penguins').dropna().reset_index(drop=True)
    # X = penguins.select_dtypes(include=['number'])
    # y = penguins['species']

    # model = SparseDecisionTree(alpha=selected_lambda)
    # model.fit(X, y)

    # # Get metrics
    # metrics = model.get_metrics(X, y)

    # # Generate visualization and convert to base64
    # image_path = 'static/tree.png'
    # model.export_tree_graphviz(X.columns, save_path=image_path)

    # with open(image_path, "rb") as image_file:
    #     img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # context.update({
    #     'lambda': selected_lambda,
    #     "test_accuracy": metrics['test_acc'],
    #     "train_accuracy": metrics['train_acc'],
    #     "depth": metrics['tree_depth'],
    #     "n_nodes": metrics['num_leaves'],
    #     "total_nodes": metrics['total_nodes'],
    #     "tree_image": img_base64,
    #     "show_tree": True,
    #     'message': f"Tree trained with lambda (alpha): {selected_lambda}"
    # })

# Train DT
    tree_model = PalmerPenguinsDecisionTree(
        max_depth = 5, 
        min_samples_split = 2, 
        random_state = 42, 
    )

    tree_model.train_model()
    metrics = tree_model.get_metrics()
    img_path = tree_model.generate_tree_visualization(save_path = 'static/tree.png')
    
    with open(img_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    context.update({
        'lambda': selected_lambda,
        "test_accuracy": metrics['test_accuracy'], 
        "depth": metrics['tree_depth'], 
        "n_nodes": metrics['num_leaves'],
        "total_nodes": metrics['total_nodes'],
        "tree_image": img_base64,
        "show_tree": True,
        'message': f"Tree trained with lambda: {selected_lambda}"
    })

    return render(request, 'project3_base.html', context)


def get_tree_depth(tree):
    def dfs(node_id, depth=0):
        node = tree[node_id]
        if "left" not in node and "right" not in node: 
            return depth
        return max(
            dfs(node['left'], depth +1), 
            dfs(node['right'], depth +1)
        )
    return dfs(0)

def logistic_regression_visualization(request, context):
    pass
    #return render(request, 'task2.html', context)

def counterfactual(request, context):
    penguins = sns.load_dataset('penguins').dropna().reset_index(drop=True)
    #print(penguins.head())
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