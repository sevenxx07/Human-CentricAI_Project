from project_2.ml_models.LogisticRegressionModel import LogisticRegressionModel
from project_2.ml_models.NaiveBayesModel import NaiveBayesModel
from project_2.ml_models.SVMModel import SVMModel
from project_2.view.views import DATA_STORAGE


def get_raw_text_for_sample(sample_idx, data_set_cache=None):
    """Get the raw text for a specific sample index"""
    # Use global cache if no specific cache provided
    if data_set_cache is None:
        from project_2.view.views_task2 import DATASET_CACHE
        data_set_cache = DATASET_CACHE

    if not data_set_cache.get('loaded', False):
        raise ValueError("Dataset not loaded")

    # Get raw texts from cache
    raw_texts = data_set_cache.get('texts', [])

    if sample_idx >= len(raw_texts):
        raise IndexError(f"Sample index {sample_idx} out of range (max: {len(raw_texts) - 1})")

    return raw_texts[sample_idx]


def create_model_from_pretrained_config(logger=None):
    """Create a NEW model instance based on pre-trained configuration - same type but fresh"""
    classifier_settings = DATA_STORAGE.get('classifier_settings')
    if not classifier_settings:
        raise ValueError("No pre-trained classifier settings found")

    model_type = classifier_settings.model_type

    if logger:
        logger.info(f"Creating new {model_type} model instance matching pre-trained configuration")

    if model_type == 'logistic':
        return LogisticRegressionModel(
            C=classifier_settings.regularization_c,
            max_iter=classifier_settings.max_iter,
            solver=classifier_settings.solver,
            penalty=classifier_settings.penalty,
            random_state=42,
            verbose=False
        )
    elif model_type == 'naive_bayes':
        return NaiveBayesModel(
            variant=classifier_settings.nb_variant,
            alpha=classifier_settings.alpha,
            fit_prior=classifier_settings.fit_prior,
            random_state=42,
            verbose=False
        )
    elif model_type == 'svm':
        return SVMModel(
            C=classifier_settings.regularization_c,
            kernel=classifier_settings.kernel,
            gamma=classifier_settings.gamma,
            max_iter=classifier_settings.max_iter,
            random_state=42,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown model type from pre-trained config: {model_type}")