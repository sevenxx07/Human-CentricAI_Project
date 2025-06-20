import random

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class ActiveLearningLoop:
    """
    Core active learning loop implementation for pool-based active learning.
    """

    def __init__(self, X, y, utility_function, model, test_size=0.2, random_state=42):
        """
        Initialize the active learning loop.

        Parameters:
        -----------
        X : array-like
            Feature vectors (all available data)
        y : array-like
            True labels (hidden during active learning, used for oracle)
        utility_function : UtilityFunction
            Instance of utility function for sample selection
        model : ClassifierWrapper
            Model instance (should be untrained)
        test_size : float
            Proportion of data to use as test set
        random_state : int
            Random state for reproducibility
        """
        self.X = X
        self.y = np.array(y)
        self.utility_function = utility_function
        self.model = model
        self.random_state = random_state

        # Split into train/test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Initialize active learning state - use shape[0] for sparse matrices
        self.labeled_indices = set()
        self.unlabeled_indices = set(range(self.X_train.shape[0]))
        self.query_history = []
        self.accuracy_history = []
        self.is_initialized = False

        print(f"DEBUG: Initialized AL loop with {self.X_train.shape[0]} training samples")

    def initialize_with_random_samples(self, n_initial=10):
        """
        Initialize the labeled set with random samples.

        Parameters:
        -----------
        n_initial : int
            Number of initial samples to label
        """
        if self.is_initialized:
            raise RuntimeError("Active learning loop already initialized")

        # Select initial samples randomly
        initial_samples = random.sample(list(self.unlabeled_indices), n_initial)

        for idx in initial_samples:
            self._label_sample(idx)

        # Train initial model
        self._train_current_model()
        self.is_initialized = True

        print(f"Initialized with {n_initial} random samples")

    def initialize_with_specific_samples(self, sample_indices):
        """
        Initialize with specific sample indices (useful for Django integration).

        Parameters:
        -----------
        sample_indices : list
            List of indices to use as initial labeled samples
        """
        if self.is_initialized:
            raise RuntimeError("Active learning loop already initialized")

        for idx in sample_indices:
            if idx in self.unlabeled_indices:
                self._label_sample(idx)

        self._train_current_model()
        self.is_initialized = True

    def get_next_query(self, n_candidates=None):
        """
        Get the next sample to query based on utility function.

        Parameters:
        -----------
        n_candidates : int, optional
            Number of candidates to consider (None = all unlabeled)

        Returns:
        --------
        query_idx : int
            Index of the sample to query next
        utility_scores : dict
            Utility scores for all candidates (for debugging)
        """
        if not self.is_initialized:
            raise RuntimeError("Must initialize before querying")

        if len(self.unlabeled_indices) == 0:
            raise RuntimeError("No unlabeled samples remaining")

        # Get unlabeled data
        unlabeled_list = list(self.unlabeled_indices)

        # Limit candidates if specified
        if n_candidates and n_candidates < len(unlabeled_list):
            candidates = random.sample(unlabeled_list, n_candidates)
        else:
            candidates = unlabeled_list

        # Get features for candidates - handle sparse matrices properly
        if hasattr(self.X_train, 'toarray'):
            # Sparse matrix - use indexing that works with sparse matrices
            X_candidates = self.X_train[candidates]
        else:
            # Dense array
            X_candidates = self.X_train[candidates]

        # Calculate utility scores
        utility_scores = self.utility_function.apply(self.model, X_candidates)

        # Find the sample with highest utility
        best_candidate_idx = np.argmax(utility_scores)
        query_idx = candidates[best_candidate_idx]

        # Create utility scores dict for return (mapped to original indices)
        scores_dict = {candidates[i]: utility_scores[i] for i in range(len(candidates))}

        return query_idx, scores_dict

    def query_sample(self, sample_idx, label=None):
        """
        Query a sample (either with oracle or manual label).

        Parameters:
        -----------
        sample_idx : int
            Index of sample to query
        label : int, optional
            Manual label (if None, uses oracle/true label)

        Returns:
        --------
        dict : Query result information
        """
        if sample_idx not in self.unlabeled_indices:
            raise ValueError(f"Sample {sample_idx} is not in unlabeled set")

        # Use oracle if no label provided
        if label is None:
            label = self.y_train[sample_idx]

        # Label the sample
        self._label_sample(sample_idx, label)

        # Retrain model
        self._train_current_model()

        # Record query
        query_info = {
            'sample_idx': sample_idx,
            'label': label,
            'n_labeled': len(self.labeled_indices),
            'n_unlabeled': len(self.unlabeled_indices),
            'accuracy': self.get_current_accuracy()
        }

        self.query_history.append(query_info)
        self.accuracy_history.append(query_info['accuracy'])

        return query_info

    def run_automatic_loop(self, n_queries, n_candidates=None):
        """
        Run the active learning loop automatically using oracle.

        Parameters:
        -----------
        n_queries : int
            Number of queries to make
        n_candidates : int, optional
            Number of candidates to consider per query

        Returns:
        --------
        list : History of query results
        """
        if not self.is_initialized:
            raise RuntimeError("Must initialize before running loop")

        results = []

        for i in range(n_queries):
            if len(self.unlabeled_indices) == 0:
                print(f"No more unlabeled samples. Stopped after {i} queries.")
                break

            # Get next query
            query_idx, _ = self.get_next_query(n_candidates)

            # Query with oracle
            result = self.query_sample(query_idx)
            results.append(result)

            print(f"Query {i + 1}: Sample {query_idx}, "
                  f"Accuracy: {result['accuracy']:.3f}, "
                  f"Labeled: {result['n_labeled']}")

        return results

    def get_current_accuracy(self):
        """Get current model accuracy on test set."""
        if not self.model.is_trained:
            return 0.0

        predictions = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)

    def get_labeled_data(self):
        """
        Get currently labeled data.

        Returns:
        --------
        X_labeled : array
            Labeled feature vectors
        y_labeled : array
            Labeled targets
        indices : list
            Original indices of labeled samples
        """
        indices = list(self.labeled_indices)

        # Handle sparse matrices properly
        if hasattr(self.X_train, 'toarray'):
            # Sparse matrix
            X_labeled = self.X_train[indices]
        else:
            # Dense array
            X_labeled = self.X_train[indices]

        y_labeled = self.y_train[indices]
        return X_labeled, y_labeled, indices

    def get_unlabeled_data(self, n_samples=None):
        """
        Get currently unlabeled data.

        Parameters:
        -----------
        n_samples : int, optional
            Maximum number of samples to return

        Returns:
        --------
        X_unlabeled : array
            Unlabeled feature vectors
        indices : list
            Original indices of unlabeled samples
        """
        indices = list(self.unlabeled_indices)
        if n_samples and n_samples < len(indices):
            indices = random.sample(indices, n_samples)

        # Handle sparse matrices properly
        if hasattr(self.X_train, 'toarray'):
            # Sparse matrix
            X_unlabeled = self.X_train[indices]
        else:
            # Dense array
            X_unlabeled = self.X_train[indices]

        return X_unlabeled, indices

    def get_status(self):
        """
        Get current status of active learning loop.

        Returns:
        --------
        dict : Status information
        """
        return {
            'is_initialized': self.is_initialized,
            'n_labeled': len(self.labeled_indices),
            'n_unlabeled': len(self.unlabeled_indices),
            'n_queries_made': len(self.query_history),
            'current_accuracy': self.get_current_accuracy() if self.is_initialized else 0.0,
            'model_trained': self.model.is_trained
        }

    def _label_sample(self, sample_idx, label=None):
        """Internal method to label a sample."""
        if label is None:
            label = self.y_train[sample_idx]

        self.labeled_indices.add(sample_idx)
        self.unlabeled_indices.discard(sample_idx)

    def _train_current_model(self):
        """Internal method to train model on current labeled data."""
        if len(self.labeled_indices) == 0:
            return

        X_labeled, y_labeled, _ = self.get_labeled_data()
        self.model.train(X_labeled, y_labeled)