from sklearn.metrics import r2_score

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Fits a model and returns its R² score.

    Args:
        model: Any scikit-learn compatible model.
        X_train, y_train, X_test, y_test: Train/test split data.

    Returns:
        float: R² score of the model.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)
