"""Train and evaluate a subscriber-churn classifier on synthetic data."""
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


def load_data():
    X, y = make_classification(
        n_samples=2000,
        n_features=12,
        n_informative=5,
        n_redundant=2,
        weights=[0.88, 0.12],
        flip_y=0.05,
        class_sep=0.8,
        random_state=SEED,
    )
    return X, y


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("== Evaluation (held-out test set) ==")
    print(f"test_rows={len(y_test)}")
    print(f"accuracy={accuracy_score(y_test, pred):.3f}")
    print(f"macro_f1={f1_score(y_test, pred, average='macro'):.3f}")


if __name__ == "__main__":
    main()
