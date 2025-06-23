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
        self.test_accuracy = None
        self.num_leaves = 0

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

        def parse_tree_node(text):
            """Recursively parse tree nodes from string representation."""
            text = text.strip()

            # Check if this is a leaf node (prediction)
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

            # Find the bracket that contains the children
            children_start = text.find('[', text.find('feature:'))
            if children_start == -1:
                return None

            # Find the matching closing bracket
            bracket_count = 1
            children_end = children_start + 1
            while children_end < len(text) and bracket_count > 0:
                if text[children_end] == '[':
                    bracket_count += 1
                elif text[children_end] == ']':
                    bracket_count -= 1
                children_end += 1

            if bracket_count > 0:
                return None

            children_content = text[children_start + 1:children_end - 1].strip()

            # Find left child and right child
            left_start = children_content.find('left child:')
            right_start = children_content.find('right child:')

            if left_start == -1 or right_start == -1:
                return None

            # Extract left child - need to find the complete block
            left_content_start = left_start + len('left child:')
            left_content = children_content[left_content_start:right_start].strip()

            # Remove trailing comma and whitespace
            left_content = left_content.rstrip(',').strip()

            # If it starts and ends with braces, remove them
            if left_content.startswith('{') and left_content.endswith('}'):
                left_content = left_content[1:-1].strip()

            # Extract right child - from right_start to end
            right_content_start = right_start + len('right child:')
            right_content = children_content[right_content_start:].strip()

            # Handle the right child which might have nested brackets
            if right_content.startswith('{'):
                # Find the matching closing brace
                brace_count = 1
                end_pos = 1
                while end_pos < len(right_content) and brace_count > 0:
                    if right_content[end_pos] == '{':
                        brace_count += 1
                    elif right_content[end_pos] == '}':
                        brace_count -= 1
                    end_pos += 1

                if brace_count == 0:
                    right_content = right_content[1:end_pos - 1].strip()

            # Recursively parse children
            left_child = parse_tree_node(left_content)
            right_child = parse_tree_node(right_content)

            return {
                'feature': feature_idx,
                'left_child': left_child,
                'right_child': right_child
            }

        # Step 1: Separate the three components from the input string

        # Find the end of the tree structure (look for the closing brace followed by }, Index)
        # The tree starts after the first { and we need to find its matching }
        tree_start = input_string.find('{ feature:')
        if tree_start == -1:
            tree_start = input_string.find('{')

        # Find the matching closing brace for the tree
        brace_count = 0
        tree_end = tree_start
        for i in range(tree_start, len(input_string)):
            if input_string[i] == '{':
                brace_count += 1
            elif input_string[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    tree_end = i + 1
                    break

        # Extract tree string
        tree_part = input_string[tree_start:tree_end]

        # Find the Index part
        index_start = input_string.find("Index([", tree_end)
        if index_start == -1:
            raise ValueError("Could not find Index in the input string")

        # Find the end of the Index (look for the closing bracket and parenthesis)
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

        # Extract number of classes (should be the remaining part)
        remaining = input_string[index_end:].strip().strip(',').strip()
        num_classes = int(remaining)

        # Step 2: Parse the tree structure
        tree_dict = parse_tree_node(tree_part)

        # Step 3: Parse the Index to extract feature names
        # Extract content between the square brackets
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
                # Check if it's escaped
                if i > 0 and features_content[i - 1] != '\\':
                    in_quotes = False
                    quote_char = None
            elif not in_quotes and char == ',':
                # End of current feature
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
        self.swap_numbers_for_text_dictionary(a, class_names, b)
        print(a)

        def build_tree(dict, par, r, counter):
            current_id = counter[0]

            for k in dict:
                if k == 'feature':
                    dot.node(str(current_id), label=dict[k])
                    if par is not None:
                        if r == 1:
                            dot.edge(str(par), str(current_id), label="NO")
                        elif r == 0:
                            dot.edge(str(par), str(current_id), label="YES")
                    counter[0] += 1
                elif k == 'prediction':
                    color = {
                        'Adelie': 'lightblue',
                        'Gentoo': 'lightgreen',
                        'Chinstrap': 'lightpink'
                    }.get(dict[k], 'white')
                    dot.node(str(current_id), label=f"{dict[k]}\nloss: {dict['loss']}", fillcolor=color, style='filled')
                    if par is not None:
                        if r == 1:
                            dot.edge(str(par), str(current_id), label="NO")
                        elif r == 0:
                            dot.edge(str(par), str(current_id), label="YES")
                    counter[0] += 1
                    return
                elif k == 'left_child':
                    build_tree(dict[k], current_id, 0, counter)
                elif k == 'right_child':
                    build_tree(dict[k], current_id, 1, counter)
        build_tree(a, None, None, [0])
        print(self.num_leaves)
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
