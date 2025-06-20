import graphviz
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from gosdt import ThresholdGuessBinarizer, GOSDTClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class SparseDecisionTree:
    def __init__(self, alpha=0.01, depth_budget=6, time_limit=60, verbose=True):
        self.alpha = alpha
        self.depth_budget = depth_budget
        self.time_limit = time_limit
        self.verbose = verbose

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_bin = None
        self.X_test_bin = None
        self.ref_labels = None
        self.clf = None
        self.label_encoder = None

    def load_data(self, test_size=0.2, random_state=42):
        df = load_penguins()
        df = df.dropna()
        y = df["species"]
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)  # Store encoded values
        X = df.drop(columns=["species"])
        X = pd.get_dummies(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def guess_thresholds(self):
        """Binarize continuous features using gradient boosting thresholds."""
        enc = ThresholdGuessBinarizer(n_estimators=40, max_depth=1, random_state=42)
        enc.set_output(transform="pandas")
        self.X_train_bin = enc.fit_transform(self.X_train, self.y_train)
        self.X_test_bin = enc.transform(self.X_test)

    def get_reference_labels(self):
        """Generate reference predictions from a black-box model."""
        ref_model = GradientBoostingClassifier(n_estimators=40, max_depth=1, random_state=42)
        ref_model.fit(self.X_train_bin, self.y_train)
        self.ref_labels = ref_model.predict(self.X_train_bin)

    def train(self):
        """Train the GOSDT sparse decision tree."""
        self.clf = GOSDTClassifier(
            regularization=self.alpha,
            similar_support=False,
            time_limit=self.time_limit,
            depth_budget=self.depth_budget,
            verbose=self.verbose
        )
        self.clf.fit(self.X_train_bin, self.y_train, y_ref=self.ref_labels)

    def evaluate(self):
        """Return accuracy on train and test data."""
        train_acc = self.clf.score(self.X_train_bin, self.y_train)
        test_acc = self.clf.score(self.X_test_bin, self.y_test)
        return train_acc, test_acc

    def run_pipeline(self):
        """Complete training pipeline."""
        self.load_data()
        self.guess_thresholds()
        self.get_reference_labels()
        self.train()
        return self.evaluate()

    def _parse_tree_recursive(self, tree_node, feature_names, depth=0, path=""):
        """
        Recursively parse the tree node and extract decision rules.
        """
        rules = []

        if hasattr(tree_node, 'prediction') and tree_node.prediction is not None:
            # Leaf node
            prediction = tree_node.prediction
            loss = getattr(tree_node, 'loss', 0)
            species_name = self.label_encoder.inverse_transform([prediction])[0]

            if path:
                rule = f"IF {path} THEN predict {species_name} (loss: {loss:.4f})"
            else:
                rule = f"predict {species_name} (loss: {loss:.4f})"
            rules.append(rule)

        elif hasattr(tree_node, 'feature') and tree_node.feature is not None:
            # Internal node
            feature_idx = tree_node.feature
            feature_name = feature_names[feature_idx]

            # Process left child (feature is True/satisfied)
            if hasattr(tree_node, 'left') and tree_node.left is not None:
                left_path = f"{path} AND {feature_name}" if path else feature_name
                rules.extend(self._parse_tree_recursive(tree_node.left, feature_names, depth + 1, left_path))

            # Process right child (feature is False/not satisfied)
            if hasattr(tree_node, 'right') and tree_node.right is not None:
                negated_condition = self._negate_condition(feature_name)
                right_path = f"{path} AND {negated_condition}" if path else negated_condition
                rules.extend(self._parse_tree_recursive(tree_node.right, feature_names, depth + 1, right_path))

        return rules

    def _negate_condition(self, condition):
        """
        Negate a condition (e.g., 'x <= 5' becomes 'x > 5').
        """
        if '<=' in condition:
            return condition.replace('<=', '>')
        elif '<' in condition:
            return condition.replace('<', '>=')
        elif '>=' in condition:
            return condition.replace('>=', '<')
        elif '>' in condition:
            return condition.replace('>', '<=')
        else:
            return f"NOT {condition}"

    def print_decision_rules(self):
        """
        Recursively parses the GOSDT tree and prints human-readable if-then rules.
        """
        if self.clf is None or not self.clf.trees_:
            raise RuntimeError("Train the model before extracting rules.")

        tree_root = self.clf.trees_[0]
        feature_names = self.X_train_bin.columns.tolist()

        print("\n" + "=" * 60)
        print("DECISION RULES")
        print("=" * 60)

        # Parse the tree structure
        rules = self._parse_tree_recursive(tree_root, feature_names)

        for i, rule in enumerate(rules, 1):
            print(f"\nRule {i}:")
            print(f"  {rule}")

        print("\n" + "=" * 60)

    def _add_nodes_to_graph(self, dot, tree_node, feature_names, node_id=0, parent_id=None, edge_label=""):
        """
        Recursively add nodes and edges to the graphviz graph.
        """
        current_id = node_id

        if hasattr(tree_node, 'prediction') and tree_node.prediction is not None:
            # Leaf node
            prediction = tree_node.prediction
            loss = getattr(tree_node, 'loss', 0)
            species_name = self.label_encoder.inverse_transform([prediction])[0]

            label = f"{species_name}\\nloss: {loss:.4f}"
            dot.node(str(current_id), label, shape='box', style='filled', fillcolor='lightblue')

            if parent_id is not None:
                dot.edge(str(parent_id), str(current_id), label=edge_label)

            return current_id + 1

        elif hasattr(tree_node, 'feature') and tree_node.feature is not None:
            # Internal node
            feature_idx = tree_node.feature
            feature_name = feature_names[feature_idx]

            # Clean up feature name for display
            display_name = feature_name.replace('_', ' ').title()
            if len(display_name) > 25:
                display_name = display_name[:22] + "..."

            dot.node(str(current_id), display_name, shape='ellipse', style='filled', fillcolor='lightgreen')

            if parent_id is not None:
                dot.edge(str(parent_id), str(current_id), label=edge_label)

            next_id = current_id + 1

            # Add left child (True/Yes)
            if hasattr(tree_node, 'left') and tree_node.left is not None:
                next_id = self._add_nodes_to_graph(dot, tree_node.left, feature_names,
                                                   next_id, current_id, "Yes")

            # Add right child (False/No)
            if hasattr(tree_node, 'right') and tree_node.right is not None:
                next_id = self._add_nodes_to_graph(dot, tree_node.right, feature_names,
                                                   next_id, current_id, "No")

            return next_id

        return current_id + 1

    def export_tree_image(self, filename="decision_tree", format="png"):
        """
        Export the decision tree as a graphical image using graphviz.
        """
        if self.clf is None or not self.clf.trees_:
            raise RuntimeError("Train the model before exporting tree image.")

        # Create a new directed graph
        dot = graphviz.Digraph(comment='Sparse Decision Tree')
        dot.attr(rankdir='TB')  # Top to bottom layout
        dot.attr('node', fontname='Arial', fontsize='10')
        dot.attr('edge', fontname='Arial', fontsize='9')

        tree_root = self.clf.trees_[0]
        print(tree_root)
        feature_names = self.X_train_bin.columns.tolist()

        # Build the graph
        self._add_nodes_to_graph(dot, tree_root, feature_names)

        # Save the file
        output_path = dot.render(filename, format=format, cleanup=True)

        return output_path

if __name__ == "__main__":
    tree_model = SparseDecisionTree(alpha=0.05)
    train_acc, test_acc = tree_model.run_pipeline()

    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")

    img_path = tree_model.export_tree_image()
    print(f"Tree image saved to: {img_path}")
    tree_model.print_decision_rules()