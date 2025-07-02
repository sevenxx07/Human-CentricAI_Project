import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class PlainLogisticRegressionModel:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_data(self, test_size=0.3, random_state=42):

        # Load the dataset
        penguins = load_penguins()
        penguins_clean = penguins.dropna()
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
        #df = load_penguins().dropna()
        #y = df["species"]
        # self.label_encoder = LabelEncoder()
        # y_encoded = self.label_encoder.fit_transform(y)

        # X = pd.get_dummies(df.drop(columns=["species"]))
        # self.feature_names = X.columns.tolist()

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        #     X, y_encoded, test_size=test_size, random_state=random_state
        # )

    def train(self):
        self.load_data()
        self.model = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000
        )
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        y_pred_numeric = self.model.predict(X)
        if self.label_encoder is not None and len(y_pred_numeric)>0:
            y_pred_label = self.label_encoder.inverse_transform(y_pred_numeric)
            return y_pred_label
        else:
            return y_pred_numeric
 
        #return self.model.predict(X)

    def evaluate(self):
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)

        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)

        print(f"Train Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        return train_acc, test_acc

    def print_coefficients(self):
        classes = self.label_encoder.classes_
        coef_matrix = self.model.coef_  # shape: (n_classes, n_features)
        # Coefficients shape: (n_classes, n_features)

        # Find non-zero coefficients
        non_zero_mask = (coef_matrix != 0)

        # Count how many features are used (at least once across any class)
        used_features = (non_zero_mask.any(axis=0)).sum()

        print(f"Used Features: {used_features} / {coef_matrix.shape[1]}")

        print("\nMODEL COEFFICIENTS:")
        for i, class_name in enumerate(classes):
            print(f"\nClass: {class_name}")
            for fname, weight in zip(self.feature_names, coef_matrix[i]):
                print(f"  {fname}: {weight:.4f}")

    def run_pipeline(self):
        self.train()
        train_acc, test_acc = self.evaluate()
        self.print_coefficients()
        return train_acc, test_acc

if __name__ == "__main__":
    logreg = PlainLogisticRegressionModel()
    logreg.run_pipeline()
    logreg.predict()