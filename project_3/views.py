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
from project_3.Interpretability.Logistic_regression_complexity import SparseLogisticRegression
from project_3.Counterfactuals.counterfactuals_workflow import CounterfactualExplainer
from project_3.Interpretability.Logistic_regression import PlainLogisticRegressionModel

# Initialize global variables for models
global_trained_DT = None
global_trained_DT_sparse = None
global_trained_LR = None
global_trained_LR_sparse = None

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

        elif action == 'logistic_regression':
            logistic_regression(request, context)

        elif action == 'sparse_logistic_regression':
            sparse_logistic_regression(request, context)

        elif action == 'counterfactual':
            counterfactual(request, context)

    return render(request, 'project3_base.html', context)

def DT(request, context):
    global global_trained_DT

    tree_model = PalmerPenguinsDecisionTree()
    tree_model.train()
    global_trained_DT = tree_model
    
    context['trained_DT'] = tree_model
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

def sparse_DT(request, context):
    global global_trained_DT_sparse
    
    selected_lambda = float(request.POST.get('lambda', 0.1))
    print('Selected lambda:', selected_lambda)

    try: 
        model = SparseDecisionTree(alpha=selected_lambda)
        test_acc = model.run_pipeline()
    except RuntimeError:
        context['error'] = (
        f"The selected sparsity value (lambda = {selected_lambda}) is too high, "
        "and the model failed to converge. Please try a lower lambda."
    )
        return render(request, "project3_base.html", context)

    global_trained_DT_sparse = model
    context['trained_DT_sparse'] = model

    image_path = model.export_tree_image()
    n_leaves = model.num_of_leaves()
    print(f"n_leaves: {n_leaves}")
    with open(image_path, "rb") as image_file:
        img_base64_sparse = base64.b64encode(image_file.read()).decode('utf-8')

    context.update({
        "train_acc": True,
        'lambda': selected_lambda,
        "test_accuracy_sparse": test_acc,
        "n_nodes_sparse": n_leaves, #Returns 0 nodes in the interface... FIX!
        "tree_image_sparse": img_base64_sparse,
        "show_tree_sparse": True,
        'message': f"Tree trained with lambda (alpha): {selected_lambda}"
    })

def logistic_regression(request, context):
    global global_trained_LR

    lr_model = PlainLogisticRegressionModel()
    
    train_acc_lr, test_acc_lr = lr_model.run_pipeline()
    print("test accuracy LR:", test_acc_lr)

    global_trained_LR = lr_model
    
    context.update({
        "trained_LR": True,
        "test_acc_lr": test_acc_lr, 
    })
    return render(request, 'project3_base.html', context)
 

def sparse_logistic_regression(request, context):
    global global_trained_LR_sparse

    selected_alpha = float(request.POST.get('alpha', 10))
    alpha = selected_alpha * 100

    lr_sparse_model = SparseLogisticRegression(alpha = alpha)

    test_acc_sparse_lr, nr_of_used_features = lr_sparse_model.run_pipeline()
    used_features, unused_features = lr_sparse_model.get_used_and_unused_features()
    #class_coeffs = lr_sparse_model.get_nonzero_coefficients()
    total_nr_of_coeffs = len(lr_sparse_model.feature_names)

    show_detailed = 'show_detailed' in request.POST
    if show_detailed:
        context['nonzero_coefficients'] = lr_sparse_model.get_nonzero_coefficients()

    print(test_acc_sparse_lr, nr_of_used_features)
    global_trained_LR_sparse = lr_sparse_model
    
    global_trained_LR_sparse = lr_sparse_model

    context.update({
        "trained_LR_sparse": True,
        #"class_coeffs": class_coeffs,
        "used_features": used_features,
        "unused_features": unused_features, 
        "show_detailed_coeff": show_detailed,
        "alpha" : alpha, 
        "test_accuracy_sparse_lr": test_acc_sparse_lr, 
        "nr_of_used_features": f"{len(used_features)} / {len(used_features) + len(unused_features)}",
        'message': f"Tree trained with alpha: {selected_alpha}"
    })
    return render(request, 'project3_base.html', context)

def counterfactual(request, context):
    global global_trained_DT, global_trained_DT_sparse, global_trained_LR, global_trained_LR_sparse
    
    data = context['dataset']
    numeric_columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    categorical_columns = ['island', 'sex']

    data_encoded = data.copy()
    for col in categorical_columns: 
        data_encoded[col] = pd.Categorical(data_encoded[col]).codes

    # Get POST input values 
    model_type = request.POST.get("model")
    actual_species = request.POST.get("actual_species")
    sample_id = request.POST.get("sample_id")
    target_label = request.POST.get("target_species")
    N = int(request.POST.get("N", 500))
    k = int(request.POST.get("k", 3))

    if not actual_species or not sample_id or not target_label or not model_type:
        context["error"] = "Please select species, sample, target & model"
        return render(request, "project3_base.html", context)
    
    try:
        sample_id = int(sample_id)
    except ValueError:
        context['error'] = "Invalid sample ID"
        return render(request, "project3_base.html", context)
    
    print(f"global_trained_DT: {global_trained_DT}")
    print(f"global_trained_DT_sparse: {global_trained_DT_sparse}")
    print(f"global_trained_LR: {global_trained_LR}")
    print(f"global_trained_LR_sparse: {global_trained_LR_sparse}")

    model_map = {
        'dt': global_trained_DT,
        'sparse_dt': global_trained_DT_sparse, 
        'lr': global_trained_LR, 
        'sparse_lr': global_trained_LR_sparse
                }
    
    model = model_map.get(model_type)
    print(f"model_type: {model_type}, model: {model}")

    
    if model is None:
        context["error"] = f"Model '{model_type}' has not been trained yet."
        return render(request, "project3_base.html", context)

    if sample_id < 0 or sample_id >= len(data):
        context["error"] = "Sample ID out of range."
        return render(request, "project3_base.html", context)

    species_to_label = {name: i for i, name in enumerate(data['species'].unique())}
    if target_label in species_to_label:
        y_target = species_to_label[target_label]
    else: 
        context["error"] = "Invalid target species."
        return render(request, "project3_base.html", context)
    
    data_for_explainer = data_encoded.drop(columns=['species', 'year'], errors='ignore')
    
    print(f"Model type: {type(model)}")
    print(f"Has predict attribute: {hasattr(model, 'predict')}")
    print(f"Model predict callable? {callable(getattr(model, 'predict', None))}")

    explainer = CounterfactualExplainer(
        model = model,
        data = data_encoded,  
        numeric_columns = numeric_columns, 
        categorical_columns = categorical_columns,
        N=N,
        k=k
    )

    x = data_for_explainer.iloc[sample_id].astype(float)
  
    counterfactuals = explainer.compute(x, y_target)
    print("Input features (x) for counterfactual computation:")
    print(x)
    print("Target label (y_target):", y_target)


    if counterfactuals: 
        counterfactual_keys = list(counterfactuals[0].keys())
    else:
        counterfactual_keys= []

    context.update({
        'counterfactuals': counterfactuals,
        'selected_example': x.to_dict(),
        'target_label': target_label,
        'N': N,
        'k': k,
        'model_type' : model_type,
        'example_index': sample_id,
        'message': f'Found {len(counterfactuals)} counterfactual explanations.'

    })

    return render(request, 'project3_base.html', context)


def get_samples(request):
    print("get_samples() view triggered")
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