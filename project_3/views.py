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
from django.http import JsonResponse
from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
from sklearn.metrics import accuracy_score 

from project_3.Interpretability.Decision_tree_complexity import SparseDecisionTree
# from project_3.Interpretability.Logistic_regression_complexity import DT


import graphviz
from gosdt import GOSDTClassifier


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

    selected_lambda = float(request.POST.get('lambda', 0.1))
    print('Selected model:', selected_lambda)

# Train DT
    tree_model = SparseDecisionTree(alpha = selected_lambda)
    train_acc, test_acc = tree_model.run_pipeline()
    #img_path = tree_model.export_tree_image()

    img_base64 = None

    #encoded_string = None
    max_depth = "Unavailable"
    n_nodes = "unavailable"

    # dot = tree_model.clf.plot()
    # graph = graphviz.Source(dot)
    # graph.format = 'png'
    # out_file = graph.render('tree_visualization', cleanup=True)

    # with open(out_file, 'rb') as f:
    #     encoded_string = base64.b64encode(f.read()).decode('utf-8')

# # Convert image to base64 
#     with open(img_path, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Extract tree stats
    try: 
        tree_structure = tree_model.clf.model_
        n_nodes = len(tree_structure)
        max_depth = get_tree_depth(tree_structure)
    
    except Exception as e: 
        print("Error extracting tree structure:", e)
        n_nodes = "Unavailable"
        max_depth = "Unavailable"

    context.update({
        'lambda': selected_lambda,
        'train_accuracy': train_acc,
        "test_accuracy": test_acc, 
        "depth": max_depth, 
        "n_nodes": n_nodes,
        #"tree_image": encoded_string,
        'tree_image': img_base64,
        "show_tree": True,
        #"show_tree": False,
        "message":"Tree visualization not yet available"
    })
    #return render(request, 'proje.html', context)

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


