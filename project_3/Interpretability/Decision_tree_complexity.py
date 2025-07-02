import graphviz
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from gosdt import ThresholdGuessBinarizer, GOSDTClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score


class SparseDecisionTree:
    def __init__(self, alpha, depth_budget=6, time_limit=60, verbose=True):
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
        self.y_pred = None
        self.ref_labels = None
        self.clf = None
        self.label_encoder = None
        self.num_leaves = 0
        self.feature_names = None

    def load_data(self, test_size=0.3, random_state=42):
        # df = load_penguins()
        # df = df.dropna()
        # y = df["species"]
        # self.label_encoder = LabelEncoder()
        # y = self.label_encoder.fit_transform(y)  # Store encoded values
        # X = df.drop(columns=["species"])
        # X = pd.get_dummies(X)
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        #     X, y, test_size=test_size, random_state=random_state
        # )

        penguins_clean = load_penguins().dropna()
        y = penguins_clean['species']
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Prepare features (X) - using numerical features
        numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        categorical_features = ['island', 'sex']
       
        X = penguins_clean[numerical_features].copy()

        for cat_feature in categorical_features:
            if cat_feature in penguins_clean.columns:
                le = LabelEncoder()
                X[cat_feature] = le.fit_transform(penguins_clean[cat_feature])

        # Store feature and target names
        self.feature_names = list(X.columns)
        self.target_names = sorted(y.unique())

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state
    )


    def guess_thresholds(self):
        """Binarize continuous features using gradient boosting thresholds."""
        enc = ThresholdGuessBinarizer(n_estimators=40, max_depth=1, random_state=42)
        enc.set_output(transform="pandas")
        self.X_train_bin = enc.fit_transform(self.X_train, self.y_train)
        self.X_test_bin = enc.transform(self.X_test)
        self.feature_names = self.X_train_bin.columns.tolist()

    def get_reference_labels(self):
        """Generate reference predictions from a black-box model."""
        ref_model = GradientBoostingClassifier(n_estimators=40, max_depth=1, random_state=42)
        ref_model.fit(self.X_train_bin, self.y_train)
        self.ref_labels = ref_model.predict(self.X_train_bin)

    def train(self):
        self.load_data()
        self.clf = GOSDTClassifier(regularization=self.alpha, depth_budget=self.depth_budget, time_limit=60, similar_support=False)
        self.guess_thresholds()
        self.clf.fit(self.X_train_bin, self.y_train)

    def hard_train(self):
        """Train the GOSDT sparse decision tree."""
        self.clf = GOSDTClassifier(
            regularization=self.alpha,
            similar_support=False,
            time_limit=self.time_limit,
            depth_budget=self.depth_budget,
            verbose=self.verbose
        )
        self.y_pred = self.clf.fit(self.X_train_bin, self.y_train, y_ref=self.ref_labels)

    def evaluate(self):
        """Return accuracy on train and test data."""
        test_acc = self.clf.score(self.X_test_bin, self.y_test)
        return test_acc

    def run_pipeline(self):
        """Complete training pipeline."""
        self.train()
        return self.evaluate()

    def num_of_leaves(self):
        return self.num_leaves

    def predict(self, X):
        """
        Predict labels for new input data.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            New samples to classify. Must match original training structure.

        Returns:
        --------
        np.ndarray of predicted class labels

        
        """
        if self.clf is None:
            raise RuntimeError("Model has not been trained yet.")

        if isinstance(X, pd.Series):
            X = X.to_frame().T  # Convert to single-row DataFrame

        elif isinstance(X, list) or isinstance(X, np.ndarray):
            if np.array(X).ndim == 1: 
                X = pd.DataFrame([X])
            else:
                X = pd.DataFrame(X)
        # Ensure same dummy encoding
        X_encoded = pd.get_dummies(X)
        print("Encoded input X_encoded:\n", X_encoded)
        missing_cols = set(self.X_train.columns) - set(X_encoded.columns)
        for col in missing_cols:
            X_encoded[col] = 0
        X_encoded = X_encoded[self.X_train.columns]  # Align column order

        # Binarize using the same encoder
        enc = ThresholdGuessBinarizer(n_estimators=40, max_depth=1, random_state=42)
        enc.set_output(transform="pandas")
        enc.fit(self.X_train, self.y_train)  # Use training fit to maintain same thresholds
        X_bin = enc.transform(X_encoded)

        y_pred_numeric = self.clf.predict(X_bin)

    # Decode numeric predictions to original labels if label_encoder exists
        if self.label_encoder is not None and len(y_pred_numeric) > 0:
            y_pred_label = self.label_encoder.inverse_transform(y_pred_numeric)
            return y_pred_label
        else:
            return y_pred_numeric

        # return self.clf.predict(X_bin)

    def parse_gosdt_string(self, input_string):
        """
        Parse a single string containing GOSDT tree, feature index, and number of classes.

        Args:
            input_string (str): Combined string with tree, Index, and number of classes

        Returns:
            tuple: (tree_dict, feature_array, num_classes)
                - tree_dict: Dictionary representation of the tree structure
                - feature_array: List of feature conditions
                - num_classes: Number of classes
        """
        import re

        def find_matching_bracket(text, start_pos, open_char='[', close_char=']'):
            """Find the position of the matching closing bracket/brace."""
            count = 1
            pos = start_pos + 1
            while pos < len(text) and count > 0:
                if text[pos] == open_char:
                    count += 1
                elif text[pos] == close_char:
                    count -= 1
                pos += 1
            return pos - 1 if count == 0 else -1

        def parse_tree_node(text):
            """Recursively parse tree nodes from string representation."""
            text = text.strip()

            # Check if this is a leaf node (prediction only)
            if 'prediction:' in text and 'feature:' not in text:
                prediction_match = re.search(r'prediction:\s*(\d+),\s*loss:\s*([\d.]+)', text)
                if prediction_match:
                    return {
                        'prediction': int(prediction_match.group(1)),
                        'loss': float(prediction_match.group(2))
                    }

            # Check if this is an internal node (feature split)
            feature_match = re.search(r'feature:\s*(\d+)', text)
            if not feature_match:
                return None

            feature_idx = int(feature_match.group(1))

            # Find the opening bracket for children
            bracket_start = text.find('[', text.find('feature:'))
            if bracket_start == -1:
                return None

            # Find the matching closing bracket
            bracket_end = find_matching_bracket(text, bracket_start, '[', ']')
            if bracket_end == -1:
                return None

            # Extract the content between brackets
            children_content = text[bracket_start + 1:bracket_end].strip()

            # Parse children using a more robust method
            left_child, right_child = parse_children(children_content)

            return {
                'feature': feature_idx,
                'left_child': left_child,
                'right_child': right_child
            }

        def parse_children(content):
            """Parse left and right children from content string."""
            content = content.strip()

            # Find 'left child:' and 'right child:' positions
            left_pos = content.find('left child:')
            right_pos = content.find('right child:')

            if left_pos == -1 or right_pos == -1:
                return None, None

            # Extract left child
            left_start = left_pos + len('left child:')
            left_content = content[left_start:right_pos].strip().rstrip(',').strip()

            # Remove outer braces if present
            if left_content.startswith('{') and left_content.endswith('}'):
                left_content = left_content[1:-1].strip()

            # Extract right child - everything after 'right child:'
            right_start = right_pos + len('right child:')
            right_content = content[right_start:].strip()

            # For right child, we need to be more careful about nested structures
            if right_content.startswith('{'):
                # Find the complete right child block
                brace_count = 0
                bracket_count = 0
                end_pos = 0

                for i, char in enumerate(right_content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                    elif char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1

                    # When all braces and brackets are balanced, we found the end
                    if brace_count == 0 and bracket_count == 0 and i > 0:
                        end_pos = i + 1
                        break

                if end_pos > 0:
                    right_content = right_content[1:end_pos - 1].strip()
                else:
                    # Fallback - remove first and last brace
                    right_content = right_content[1:-1].strip() if right_content.endswith('}') else right_content[
                                                                                                    1:].strip()

            # Recursively parse both children
            left_child = parse_tree_node(left_content) if left_content else None
            right_child = parse_tree_node(right_content) if right_content else None

            return left_child, right_child

        # Step 1: Separate the three components from the input string

        # Find the tree part - starts after the colon and ends before }, Index
        tree_start = input_string.find(': {')
        if tree_start == -1:
            tree_start = input_string.find('{')
        else:
            tree_start += 2  # Skip ': '

        # Find the end of tree - look for }, Index or }, followed by newline and Index
        tree_end_pattern = r'},\s*Index\('
        tree_end_match = re.search(tree_end_pattern, input_string[tree_start:])

        if tree_end_match:
            tree_end = tree_start + tree_end_match.start() + 1  # Include the closing brace
        else:
            # Fallback - find the first }, that's followed by something that looks like Index
            pos = tree_start
            brace_count = 0
            while pos < len(input_string):
                if input_string[pos] == '{':
                    brace_count += 1
                elif input_string[pos] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Check if this is followed by something that looks like Index
                        remaining = input_string[pos:pos + 20]
                        if ', Index(' in remaining or ',\nIndex(' in remaining or ', \nIndex(' in remaining:
                            tree_end = pos + 1
                            break
                pos += 1
            else:
                tree_end = len(input_string)

        # Extract tree string
        tree_part = input_string[tree_start:tree_end].strip()

        # Find the Index part
        index_start = input_string.find("Index([", tree_end)
        if index_start == -1:
            raise ValueError("Could not find Index in the input string")

        # Find the end of the Index
        paren_count = 0
        bracket_count = 0
        index_end = index_start

        for i in range(index_start, len(input_string)):
            if input_string[i] == '(':
                paren_count += 1
            elif input_string[i] == ')':
                paren_count -= 1
            elif input_string[i] == '[':
                bracket_count += 1
            elif input_string[i] == ']':
                bracket_count -= 1

            if paren_count == 0 and bracket_count == 0 and i > index_start + 6:
                index_end = i + 1
                break

        # Extract index string
        index_part = input_string[index_start:index_end]

        # Extract number of classes
        remaining = input_string[index_end:].strip().strip(',').strip()
        num_classes = int(remaining)

        # Step 2: Parse the tree structure
        tree_dict = parse_tree_node(tree_part)

        # Step 3: Parse the Index to extract feature names
        features_match = re.search(r"Index\(\[(.*?)\],", index_part, re.DOTALL)
        if not features_match:
            raise ValueError("Could not parse feature index")

        features_content = features_match.group(1)

        # Split by comma and clean up each feature name
        feature_array = []
        current_feature = ""
        in_quotes = False
        quote_char = None

        i = 0
        while i < len(features_content):
            char = features_content[i]

            if not in_quotes and (char == "'" or char == '"'):
                in_quotes = True
                quote_char = char
            elif in_quotes and char == quote_char:
                if i > 0 and features_content[i - 1] != '\\':
                    in_quotes = False
                    quote_char = None
            elif not in_quotes and char == ',':
                feature = current_feature.strip().strip("'\"")
                if feature:
                    feature_array.append(feature)
                current_feature = ""
                i += 1
                continue

            current_feature += char
            i += 1

        # Add the last feature
        feature = current_feature.strip().strip("'\"")
        if feature:
            feature_array.append(feature)

        return tree_dict, feature_array, num_classes

    def swap_numbers_for_text_dictionary(self, dict, classes, features):
        if dict is None:
            return
        for k in dict:
            if k == 'feature':
                n = dict[k]
                dict[k] = features[n]
            if k == 'prediction':
                n = dict[k]
                dict[k] = classes[n]
                self.num_leaves += 1
                return
            if k == 'left_child' or k == 'right_child':
                self.swap_numbers_for_text_dictionary(dict[k], classes, features)
        return

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
        a, b, c = self.parse_gosdt_string(str(tree_root))
        predicted_labels = [0, 1, 2]
        class_names = self.label_encoder.inverse_transform(predicted_labels)
        print(a)
        print(b)
        print(c)
        self.swap_numbers_for_text_dictionary(a, class_names, b)
        print(a)

        def build_tree(dict, par, r, counter):
            current_id = counter[0]
            if dict is None:
                return
            for k in dict:
                if k == 'feature':
                    dot.node(str(current_id), label=dict[k])
                    if par is not None:
                        if r == 1:
                            dot.edge(str(par), str(current_id), label="False")
                        elif r == 0:
                            dot.edge(str(par), str(current_id), label="True")
                    counter[0] += 1
                elif k == 'prediction':
                    color = {
                        'Adelie': 'orange',
                        'Gentoo': 'purple',
                        'Chinstrap': 'lightgreen'
                    }.get(dict[k], 'white')
                    dot.node(str(current_id), label=f"{dict[k]}\nloss: {dict['loss']}", fillcolor=color, style='filled')
                    if par is not None:
                        if r == 1:
                            dot.edge(str(par), str(current_id), label="False")
                        elif r == 0:
                            dot.edge(str(par), str(current_id), label="True")
                    counter[0] += 1
                    return
                elif k == 'left_child':
                    build_tree(dict[k], current_id, 0, counter)
                elif k == 'right_child':
                    build_tree(dict[k], current_id, 1, counter)
        build_tree(a, None, None, [0])
        # Save the file
        output_path = dot.render(filename, format=format, cleanup=True)

        return output_path

if __name__ == "__main__":
    # Initialize and train the model
    tree_model = SparseDecisionTree(alpha=0.04) #from 0.04-0.4 so it can be optimized
    test_acc = tree_model.run_pipeline()
    print(f"Test Accuracy: {test_acc:.3f}")

    # Export the visualization
    img_path = tree_model.export_tree_image()
    print(f"Tree image saved to: {img_path}")
    print(f"Number of leaves: {tree_model.num_of_leaves()}")

    # --- Test predict() ---
    print("\n--- Testing predict() method ---")
    sample = tree_model.X_test.iloc[0]
    true_label = tree_model.y_test[0]
    predicted_label = tree_model.predict(sample)[0]
    true_name = tree_model.label_encoder.inverse_transform([true_label])[0]
    predicted_name = tree_model.label_encoder.inverse_transform([predicted_label])[0]

    print("Sample input:")
    print(sample)
    print(f"\nTrue Label: {true_name} ({true_label})")
    print(f"Predicted Label: {predicted_name} ({predicted_label})")

#for alpha in [0.004, 0.01, 0.1, 0.4]:
    #    model = SparseDecisionTree(alpha=alpha)
    #    train_acc, test_acc = model.run_pipeline()
    #    print(f"λ={alpha:.3f} → Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")