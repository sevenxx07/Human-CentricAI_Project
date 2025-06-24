import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from palmerpenguins import load_penguins
import warnings

warnings.filterwarnings('ignore')


class PalmerPenguinsDecisionTree:
    """
    A class to handle Palmer Penguins dataset analysis with Decision Tree classification.
    Provides tree visualization and performance metrics.
    """

    def __init__(self, max_depth=5, min_samples_split=2, random_state=42):
        """
        Initialize the decision tree classifier.

        Parameters:
        -----------
        max_depth : int, default=5
            Maximum depth of the tree
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node
        random_state : int, default=42
            Random state for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_names = None
        self.test_accuracy = None
        self.num_leaves = None

    def load_and_prepare_data(self):
        """
        Load Palmer Penguins dataset and prepare it for training.

        Returns:
        --------
        tuple: (X, y, feature_names, target_names)
        """
        # Load the dataset
        penguins = load_penguins()

        # Remove rows with missing values
        penguins_clean = penguins.dropna()

        # Prepare features (X) - using numerical features
        numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        X = penguins_clean[numerical_features].copy()

        # Encode categorical features if you want to include them
        categorical_features = ['island', 'sex']
        label_encoders = {}

        for cat_feature in categorical_features:
            if cat_feature in penguins_clean.columns:
                le = LabelEncoder()
                X[cat_feature] = le.fit_transform(penguins_clean[cat_feature])
                label_encoders[cat_feature] = le

        # Target variable
        y = penguins_clean['species']

        # Store feature and target names
        self.feature_names = list(X.columns)
        self.target_names = sorted(y.unique())

        return X, y, self.feature_names, self.target_names

    def train_model(self, test_size=0.3):
        """
        Train the decision tree model.

        Parameters:
        -----------
        test_size : float, default=0.3
            Proportion of dataset to include in the test split
        """
        # Load and prepare data
        X, y, feature_names, target_names = self.load_and_prepare_data()

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Initialize and train the model
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )

        # Fit the model
        self.model.fit(self.X_train, self.y_train)

        # Calculate metrics
        y_pred = self.model.predict(self.X_test)
        self.test_accuracy = accuracy_score(self.y_test, y_pred)
        self.num_leaves = int(self.model.get_n_leaves())

    def get_metrics(self):
        """
        Get model performance metrics.

        Returns:
        --------
        dict: Dictionary containing accuracy, number of leaves, and tree depth
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        return {
            'test_accuracy': self.test_accuracy,
            'num_leaves': self.num_leaves,
            'tree_depth': self.model.get_depth(),
            'total_nodes': self.model.tree_.node_count
        }

    def generate_tree_visualization(self, figsize=(20, 12), save_path='decision_tree.png', dpi=300):
        """
        Generate and save a visualization of the decision tree.

        Parameters:
        -----------
        figsize : tuple, default=(20, 12)
            Figure size (width, height) in inches
        save_path : str, default='decision_tree.png'
            Path to save the visualization
        dpi : int, default=300
            Resolution of the saved image
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        # Create the plot
        plt.figure(figsize=figsize)

        # Plot the tree
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.target_names,
            filled=True,
            rounded=True,
            fontsize=17,
            max_depth=5,  # Limit display depth for readability
            impurity = False,  # Don't show gini
            proportion = False,  # Don't show proportions
            label = 'none'  # Don't show default labels
        )

        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')

        return save_path


def main():

    # Initialize the classifier
    dt_classifier = PalmerPenguinsDecisionTree(
        max_depth=5,
        min_samples_split=2,
        random_state=42
    )
    # Train the model
    dt_classifier.train_model(test_size=0.3)
    # Generate tree visualization
    dt_classifier.generate_tree_visualization(
        figsize=(24, 16),
        save_path='palmer_penguins_decision_tree.png',
        dpi=400
    )
    met = dt_classifier.get_metrics()
    print(met)


if __name__ == "__main__":
    # Run main analysis
    main()
