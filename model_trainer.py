import pandas as pd
# Import the function for splitting data into train and test sets
from sklearn.model_selection import train_test_split
# Import the StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler
# Import our models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# Import the accuracy metric for evaluation
from sklearn.metrics import accuracy_score
# Import joblib for saving our models and scaler
import joblib
# Import LabelEncoder and numpy for previous steps
from sklearn.preprocessing import LabelEncoder
import numpy as np



# Define the path to our dataset
CSV_PATH = "features.csv"

# Load the dataset from the CSV file into a pandas DataFrame
try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")

    X = features_df.drop('genre_label', axis=1)
    y = features_df['genre_label']
    print("\n--- Label Encoding Step ---")
    
    # Check the data type of the target variable y
    # np.issubdtype checks if a dtype is a subtype of another (e.g., is int64 a subtype of integer?)
    if np.issubdtype(y.dtype, np.integer):
        print("Labels are already numerically encoded. No action needed.")
    else:
        # This block would run if your labels were strings (e.g., 'blues', 'classical')
        print("Labels are not numerical. Applying LabelEncoder.")
        
    print("\nVerification of target variable 'y':")
    print(f"Data type of y: {y.dtype}")
    print("First 5 labels:")
    print(y.head())

    print("\nVerifying the separation:")
    print(f"Shape of features (X): {X.shape}")
    print(f"Shape of target (y): {y.shape}")

    print("\nFirst 5 rows of features (X):")
    print(X.head())

    print("\nFirst 5 values of target (y):")
    print(y.head())
    # Split the data into training and testing sets
    print("\n--- Splitting Data into Training and Testing Sets ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print("\n--- Scaling Features ---")
    
    scaler = StandardScaler()
    scaler.fit(X_train)

    # --- Verification Step ---
    print("StandardScaler has been fitted to the training data.")
    # We can inspect the learned means to verify. This will be an array of 28 means.
    print(f"Learned means (μ) for the first 5 features: {scaler.mean_[:5]}")
    print(f"Learned standard deviations (σ) for the first 5 features: {scaler.scale_[:5]}")
    # --- Verification Step ---
    print("\nVerifying the shapes of the new sets:")
    print(f"Shape of X_train: {X_train.shape}") # Should be (7500, 28)
    print(f"Shape of X_test: {X_test.shape}")   # Should be (2500, 28)
    print(f"Shape of y_train: {y_train.shape}") # Should be (7500,)
    print(f"Shape of y_test: {y_test.shape}")   # Should be (2500,)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Verification Step ---
    # Let's check the properties of the newly scaled data to confirm the transformation.
    print("\nVerification of scaled data:")
    
    # The mean of each feature in the scaled training data should be very close to 0.
    print(f"Mean of first 5 features in X_train_scaled: {X_train_scaled[:, :5].mean(axis=0)}")
    
    # The standard deviation of each feature in the scaled training data should be very close to 1.
    print(f"Standard deviation of first 5 features in X_train_scaled: {X_train_scaled[:, :5].std(axis=0)}")

    # Note: The mean and std of X_test_scaled will NOT be exactly 0 and 1,
    # because it was scaled using the parameters from X_train. This is correct!
    print(f"\nMean of first 5 features in X_test_scaled: {X_test_scaled[:, :5].mean(axis=0)}")
    print("\n--- Training Logistic Regression Model ---")

    # 1. Initialize the model
    # We set 'max_iter' to a higher value to ensure the model has enough iterations
    # to converge on a solution, which helps avoid convergence warnings.
    log_reg = LogisticRegression(max_iter=1000)

    # 2. Train the model using the .fit() method
    # The model learns the relationship between our scaled training features (X_train_scaled)
    # and the corresponding genre labels (y_train).
    log_reg.fit(X_train_scaled, y_train)

    # --- Verification Step ---
    print("Logistic Regression model trained successfully!")
    
    # The trained model now contains learned parameters. For example, we can see the 10
    # classes it has learned to identify.
    print(f"Model learned the following classes: {log_reg.classes_}")
    print("\n--- Training Support Vector Machine (SVM) Model ---")

    # 1. Initialize the model
    # We use the RBF kernel, which is effective for non-linear relationships.
    # C=1.0 is the regularization parameter, a good default starting point.
    # random_state=42 ensures reproducibility.
    # probability=True is needed to enable probability estimates for later evaluation.
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)

    # 2. Train the model using the .fit() method
    # The model learns the optimal hyperplane to separate the genres based on
    # our scaled training features and their corresponding labels.
    svm_model.fit(X_train_scaled, y_train)

    # --- Verification Step ---
    print("Support Vector Machine model trained successfully!")
    print(f"Model learned the following classes: {svm_model.classes_}")
    print("\n--- Training Random Forest Classifier Model ---")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # 2. Train the model using the .fit() method
    # The model builds 100 decision trees, each learning from a different
    # bootstrap sample of the training data.
    rf_model.fit(X_train_scaled, y_train)

    # --- Verification Step ---
    print("Random Forest Classifier model trained successfully!")
    print(f"Model learned the following classes: {rf_model.classes_}")
    
    print("\n--- Evaluating Models on the Test Set ---")

    y_pred_log_reg = log_reg.predict(X_test_scaled)
    
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    print(f"Logistic Regression Accuracy: {accuracy_log_reg * 100:.2f}%")

    y_pred_svm = svm_model.predict(X_test_scaled)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Support Vector Machine Accuracy: {accuracy_svm * 100:.2f}%")
    accuracy_rf = rf_model.score(X_test_scaled, y_test)
    print(f"Random Forest Classifier Accuracy: {accuracy_rf * 100:.2f}%")
    
    print("\n--- Saving Models and Scaler to Disk ---")

    # The first argument is the Python object to save.
    # The second argument is the desired filename.
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(log_reg, 'logistic_regression_model.joblib')
    joblib.dump(svm_model, 'svm_model.joblib')
    joblib.dump(rf_model, 'random_forest_model.joblib')

    print("Scaler and models have been successfully saved to disk.")
    print("The following files have been created in your project directory:")
    print("- scaler.joblib")
    print("- logistic_regression_model.joblib")
    print("- svm_model.joblib")
    print("- random_forest_model.joblib")


except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")