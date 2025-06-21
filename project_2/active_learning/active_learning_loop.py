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

        # Termination conditions
        self.termination_conditions = {
            'type': None,  # 'accuracy', 'queries', 'budget'
            'target_accuracy': 0.85,
            'max_queries': 100,
            'budget_percent': 10,
            'is_terminated': False,
            'termination_reason': None
        }

        print(f"DEBUG: Initialized AL loop with {self.X_train.shape[0]} training samples")

    def set_termination_conditions(self, termination_type, **kwargs):
        """
        Set termination conditions for the active learning loop.

        Parameters:
        -----------
        termination_type : str
            Type of termination ('accuracy', 'queries', 'budget')
        **kwargs : dict
            Termination parameters (target_accuracy, max_queries, budget_percent)
        """
        self.termination_conditions['type'] = termination_type
        self.termination_conditions['is_terminated'] = False
        self.termination_conditions['termination_reason'] = None

        if 'target_accuracy' in kwargs:
            self.termination_conditions['target_accuracy'] = float(kwargs['target_accuracy'])
        if 'max_queries' in kwargs:
            self.termination_conditions['max_queries'] = int(kwargs['max_queries'])
        if 'budget_percent' in kwargs:
            self.termination_conditions['budget_percent'] = float(kwargs['budget_percent'])

        print(f"DEBUG: Set termination condition: {termination_type} with params {kwargs}")

    def check_termination_conditions(self):
        """
        Check if termination conditions are met.

        Returns:
        --------
        bool : True if termination conditions are met
        """
        if self.termination_conditions['type'] is None:
            return False

        current_accuracy = self.get_current_accuracy()
        n_queries = len(self.query_history)
        n_labeled = len(self.labeled_indices)
        total_training_samples = self.X_train.shape[0]
        budget_used_percent = (n_labeled / total_training_samples) * 100

        if self.termination_conditions['type'] == 'accuracy':
            if current_accuracy >= self.termination_conditions['target_accuracy']:
                self.termination_conditions['is_terminated'] = True
                self.termination_conditions[
                    'termination_reason'] = f"Target accuracy {self.termination_conditions['target_accuracy']:.3f} reached (current: {current_accuracy:.3f})"
                return True

        elif self.termination_conditions['type'] == 'queries':
            if n_queries >= self.termination_conditions['max_queries']:
                self.termination_conditions['is_terminated'] = True
                self.termination_conditions[
                    'termination_reason'] = f"Maximum queries {self.termination_conditions['max_queries']} reached"
                return True

        elif self.termination_conditions['type'] == 'budget':
            if budget_used_percent >= self.termination_conditions['budget_percent']:
                self.termination_conditions['is_terminated'] = True
                self.termination_conditions[
                    'termination_reason'] = f"Budget {self.termination_conditions['budget_percent']:.1f}% of dataset used (current: {budget_used_percent:.1f}%)"
                return True

        return False

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

    def get_next_query(self, n_candidates=None, batch_size=1, diversity_method='top_k'):
        """
        Get the next sample(s) to query based on utility function.

        Parameters:
        -----------
        n_candidates : int, optional
            Number of candidates to consider (None = all unlabeled)
        batch_size : int
            Number of samples to select (1 for single query, >1 for batch)
        diversity_method : str
            Method for batch selection ('top_k', 'diverse', 'uncertainty_diverse')

        Returns:
        --------
        query_idx or query_indices : int or list
            Index of sample (batch_size=1) or list of indices (batch_size>1)
        utility_scores : dict
            Utility scores for all candidates (for debugging)
        """
        if batch_size > 1:
            return self.get_next_batch_query(batch_size, diversity_method)

        # Single query (original behavior)
        if not self.is_initialized:
            raise RuntimeError("Must initialize before querying")

        if len(self.unlabeled_indices) == 0:
            raise RuntimeError("No unlabeled samples remaining")

        if self.termination_conditions['is_terminated']:
            raise RuntimeError(f"Active learning terminated: {self.termination_conditions['termination_reason']}")

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

    def get_next_batch_query(self, batch_size=1, diversity_method='top_k'):
        """
        Get the next batch of samples to query based on utility function.
        Updated with faster diversity methods.
        """
        if not self.is_initialized:
            raise RuntimeError("Must initialize before querying")

        if len(self.unlabeled_indices) == 0:
            raise RuntimeError("No unlabeled samples remaining")

        if self.termination_conditions['is_terminated']:
            raise RuntimeError(f"Active learning terminated: {self.termination_conditions['termination_reason']}")

        # Get unlabeled data
        unlabeled_list = list(self.unlabeled_indices)
        actual_batch_size = min(batch_size, len(unlabeled_list))

        # Get features for candidates
        if hasattr(self.X_train, 'toarray'):
            X_candidates = self.X_train[unlabeled_list]
        else:
            X_candidates = self.X_train[unlabeled_list]

        # Calculate utility scores
        utility_scores = self.utility_function.apply(self.model, X_candidates)

        # Select batch based on method
        if diversity_method == 'top_k':
            # Simply take top K samples by utility
            top_indices = np.argsort(utility_scores)[-actual_batch_size:]
            query_indices = [unlabeled_list[i] for i in top_indices]
        elif diversity_method == 'diverse_random':
            # Use weighted random selection (fastest)
            query_indices = self._select_diverse_batch_random(
                unlabeled_list, X_candidates, utility_scores, actual_batch_size
            )
        else:
            raise ValueError(f"Unknown diversity method: {diversity_method}")

        # Create utility scores dict for return
        scores_dict = {unlabeled_list[i]: utility_scores[i] for i in range(len(unlabeled_list))}

        return query_indices, scores_dict

    def _select_diverse_batch(self, candidates, X_candidates, utility_scores, batch_size):
        """
        Select a diverse batch of samples using greedy selection.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        if batch_size == 1:
            best_idx = np.argmax(utility_scores)
            return [candidates[best_idx]]

        selected_indices = []
        selected_features = []

        # Start with the highest utility sample
        first_idx = np.argmax(utility_scores)
        selected_indices.append(candidates[first_idx])

        if hasattr(X_candidates, 'toarray'):
            selected_features.append(X_candidates[first_idx].toarray().flatten())
        else:
            selected_features.append(X_candidates[first_idx].flatten())

        # Greedily select remaining samples
        for _ in range(batch_size - 1):
            remaining_candidates = [i for i, c in enumerate(candidates) if c not in selected_indices]
            if not remaining_candidates:
                break

            best_score = -float('inf')
            best_candidate = None

            for candidate_idx in remaining_candidates:
                # Get candidate features
                if hasattr(X_candidates, 'toarray'):
                    candidate_features = X_candidates[candidate_idx].toarray().flatten()
                else:
                    candidate_features = X_candidates[candidate_idx].flatten()

                # Calculate diversity score (minimum similarity to selected samples)
                similarities = [cosine_similarity([candidate_features], [sf])[0][0]
                                for sf in selected_features]
                min_similarity = min(similarities) if similarities else 0
                diversity_score = 1 - min_similarity

                # Combine utility and diversity
                combined_score = utility_scores[candidate_idx] + 0.5 * diversity_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate_idx

            if best_candidate is not None:
                selected_indices.append(candidates[best_candidate])
                if hasattr(X_candidates, 'toarray'):
                    selected_features.append(X_candidates[best_candidate].toarray().flatten())
                else:
                    selected_features.append(X_candidates[best_candidate].flatten())

        return selected_indices

    def query_sample(self, sample_idx, label=None, batch_mode=False):
        """
        Query a sample (either with oracle or manual label).

        Parameters:
        -----------
        sample_idx : int or list
            Index of sample to query (or list for batch mode)
        label : int or list, optional
            Manual label (if None, uses oracle/true label)
        batch_mode : bool
            Whether this is a batch query

        Returns:
        --------
        dict : Query result information
        """
        # Handle batch queries
        if isinstance(sample_idx, list) or batch_mode:
            return self.query_batch(sample_idx, label)

        # Single sample query
        if sample_idx not in self.unlabeled_indices:
            raise ValueError(f"Sample {sample_idx} is not in unlabeled set")

        if self.termination_conditions['is_terminated']:
            raise RuntimeError(f"Active learning terminated: {self.termination_conditions['termination_reason']}")

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
            'accuracy': self.get_current_accuracy(),
            'batch_size': 1
        }

        self.query_history.append(query_info)
        self.accuracy_history.append(query_info['accuracy'])

        # Check termination conditions after this query
        is_terminated = self.check_termination_conditions()
        query_info['is_terminated'] = is_terminated
        query_info['termination_reason'] = self.termination_conditions.get('termination_reason')

        return query_info

    def query_batch(self, sample_indices, labels=None):
        """
        Query a batch of samples (either with oracle or manual labels).

        Parameters:
        -----------
        sample_indices : list
            Indices of samples to query
        labels : list, optional
            Manual labels (if None, uses oracle/true labels)

        Returns:
        --------
        dict : Batch query result information
        """
        if not all(idx in self.unlabeled_indices for idx in sample_indices):
            invalid_indices = [idx for idx in sample_indices if idx not in self.unlabeled_indices]
            raise ValueError(f"Samples {invalid_indices} are not in unlabeled set")

        if self.termination_conditions['is_terminated']:
            raise RuntimeError(f"Active learning terminated: {self.termination_conditions['termination_reason']}")

        # Use oracle if no labels provided
        if labels is None:
            labels = [self.y_train[idx] for idx in sample_indices]

        # Label all samples in the batch
        for idx, label in zip(sample_indices, labels):
            self._label_sample(idx, label)

        # Retrain model once for the entire batch
        self._train_current_model()

        # Record batch query
        current_accuracy = self.get_current_accuracy()

        batch_query_info = {
            'sample_indices': sample_indices,
            'labels': labels,
            'batch_size': len(sample_indices),
            'n_labeled': len(self.labeled_indices),
            'n_unlabeled': len(self.unlabeled_indices),
            'accuracy': current_accuracy
        }

        # Add to history (one entry per batch)
        self.query_history.append(batch_query_info)
        self.accuracy_history.append(current_accuracy)

        # Check termination conditions after this batch
        is_terminated = self.check_termination_conditions()
        batch_query_info['is_terminated'] = is_terminated
        batch_query_info['termination_reason'] = self.termination_conditions.get('termination_reason')

        return batch_query_info

    def _select_diverse_batch_random(self, candidates, X_candidates, utility_scores, batch_size):
        """
        Fast diverse selection: weighted random sampling from top candidates.
        Fastest option - O(n log n) for sorting only.
        """
        if batch_size == 1:
            best_idx = np.argmax(utility_scores)
            return [candidates[best_idx]]

        # Take top candidates (more than batch_size for diversity)
        top_k = min(batch_size * 5, len(candidates))  # Consider 5x batch_size top candidates
        top_indices = np.argsort(utility_scores)[-top_k:]

        # Weighted random selection from top candidates
        top_utilities = utility_scores[top_indices]
        # Normalize utilities to probabilities
        probabilities = top_utilities / np.sum(top_utilities)

        # Sample without replacement
        selected_indices = np.random.choice(
            top_indices,
            size=min(batch_size, len(top_indices)),
            replace=False,
            p=probabilities
        )

        return [candidates[i] for i in selected_indices]

    def run_automatic_loop(self, n_queries, n_candidates=None, batch_size=1, diversity_method='top_k'):
        """
        Run the active learning loop automatically using oracle.

        Parameters:
        -----------
        n_queries : int
            Maximum number of queries to make (or batches if batch_size > 1)
        n_candidates : int, optional
            Number of candidates to consider per query
        batch_size : int
            Number of samples per query (1 for single, >1 for batch)
        diversity_method : str
            Method for batch selection

        Returns:
        --------
        list : History of query results
        """
        if not self.is_initialized:
            raise RuntimeError("Must initialize before running loop")

        results = []

        for i in range(n_queries):
            # Check termination conditions before making query
            if self.check_termination_conditions():
                print(f"Termination condition met: {self.termination_conditions['termination_reason']}")
                break

            if len(self.unlabeled_indices) == 0:
                print(f"No more unlabeled samples. Stopped after {i} queries.")
                break

            # Get next query (single or batch)
            try:
                if batch_size > 1:
                    query_indices, _ = self.get_next_batch_query(batch_size, diversity_method)
                    result = self.query_batch(query_indices)
                    print(f"Batch {i + 1}: Samples {query_indices}, "
                          f"Accuracy: {result['accuracy']:.3f}, "
                          f"Labeled: {result['n_labeled']}")
                else:
                    query_idx, _ = self.get_next_query(n_candidates)
                    result = self.query_sample(query_idx)
                    print(f"Query {i + 1}: Sample {query_idx}, "
                          f"Accuracy: {result['accuracy']:.3f}, "
                          f"Labeled: {result['n_labeled']}")

                results.append(result)

            except RuntimeError as e:
                print(f"Cannot get next query: {e}")
                break

            # Check if terminated after this query
            if result.get('is_terminated', False):
                print(f"Termination condition met: {result.get('termination_reason')}")
                break

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
        total_training_samples = self.X_train.shape[0]
        n_labeled = len(self.labeled_indices)
        budget_used_percent = (n_labeled / total_training_samples) * 100 if total_training_samples > 0 else 0

        status = {
            'is_initialized': self.is_initialized,
            'n_labeled': n_labeled,
            'n_unlabeled': len(self.unlabeled_indices),
            'n_queries_made': len(self.query_history),
            'current_accuracy': self.get_current_accuracy() if self.is_initialized else 0.0,
            'model_trained': self.model.is_trained,
            'budget_used_percent': budget_used_percent,
            'termination': self.termination_conditions.copy()
        }

        return status

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