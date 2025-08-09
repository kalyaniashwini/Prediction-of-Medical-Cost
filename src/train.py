import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from preprocessing import load_and_preprocess
from models import train_and_evaluate

# Paths
DATA_PATH = os.path.join("data", "insurance.csv")
RESULTS_PATH = os.path.join("results", "model_performance.txt")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

# Load data
data, _ = load_and_preprocess(DATA_PATH)

X = data.drop(columns=['charges'])
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=0),
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(),
    "SVR": SVR()
}

# Train and save results
with open(RESULTS_PATH, "w") as f:
    for name, model in models.items():
        score = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        print(f"{name} R² Score: {score:.4f}")
        f.write(f"{name}: {score:.4f}\n")
