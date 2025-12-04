from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import joblib

from Data import load_phishing_dataset

# split train/val/test sets
def split_dataset(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split X, y into train/val/test sets."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1.0 - train_ratio),
        random_state=random_state, stratify=y
    )

    val_ratio_adj = val_ratio / (val_ratio + test_ratio)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_ratio_adj),
        random_state=random_state, stratify=y_temp
    )

    print("Dataset split:")
    print("  Train:", X_train.shape)
    print("  Val:", X_val.shape)
    print("  Test:", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

# build an xgboost classifier
def build_xgb_model(n_estimators: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

def select_best_model(
    X_train, y_train, X_val, y_val,
    candidate_n_estimators: List[int],
) -> Tuple[XGBClassifier, int, float]:
    best_model, best_val_auc, best_n = None, -1.0, None

    for n in candidate_n_estimators:
        print(f"\n>>> n_estimators = {n}")

        model = build_xgb_model(n)
        model.fit(X_train, y_train)

        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)
        print(f"Validation AUC = {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model
            best_n = n
            print(f"Updated best: n={best_n}, AUC={best_val_auc:.4f}")

    print("\n==============================")
    print(f"Best n_estimators = {best_n}")
    print(f"Best validation AUC = {best_val_auc:.4f}")
    print("==============================")

    return best_model, best_n, best_val_auc

# run model evaluation on test set
def evaluate_on_test(model: XGBClassifier, X_test, y_test) -> float:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Test Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Test Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    test_auc = roc_auc_score(y_test, y_prob)
    print(f"\nTest ROC-AUC: {test_auc:.4f}")

    return test_auc


def save_model(model: XGBClassifier, model_output_path: str):
    joblib.dump(model, model_output_path)
    print(f"Model saved to: {model_output_path}")


def main():
    csv_path = "C:/Users/styu0/OneDrive/Desktop/25 full/privacy/project/dataset.csv"
    model_output_path = "C:/Users/styu0/OneDrive/Desktop/25 full/privacy/project/models/xgb_phishing_model.joblib"

    X, y = load_phishing_dataset(csv_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    candidate_n_estimators = [50, 100, 150, 200, 300, 400, 500]
    best_model, best_n, best_val_auc = select_best_model(
        X_train, y_train, X_val, y_val, candidate_n_estimators
    )

    evaluate_on_test(best_model, X_test, y_test)
    save_model(best_model, model_output_path)


if __name__ == "__main__":
    main()


