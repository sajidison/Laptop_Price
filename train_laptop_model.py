"""Train and compare laptop price prediction models, save the best one."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA = "/mnt/user-data/outputs/laptop_dataset.csv"
OUT = "/mnt/user-data/outputs"

df = pd.read_csv(DATA)
target = "price_usd"
X, y = df.drop(columns=target), df[target]
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")


def linear_prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])


def tree_prep():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols),
    ])


def wrap(prep, model):
    # Predict log(price): prices are multiplicative in nature, so this fits much better
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", prep), ("model", model)]),
        func=np.log1p, inverse_func=np.expm1)


models = {
    "Ridge Regression": wrap(linear_prep(), Ridge(alpha=1.0)),
    "Random Forest": wrap(tree_prep(), RandomForestRegressor(
        n_estimators=120, min_samples_leaf=3, max_features=0.6, n_jobs=-1, random_state=42)),
    "Gradient Boosting": wrap(tree_prep(), HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.08, max_leaf_nodes=48, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=42)),
}

results, fitted = {}, {}
for name, m in models.items():
    print(f"Training {name} ...")
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    results[name] = {
        "R2": round(r2_score(y_test, p), 4),
        "MAE_usd": round(mean_absolute_error(y_test, p), 2),
        "RMSE_usd": round(float(np.sqrt(mean_squared_error(y_test, p))), 2),
        "MAPE_pct": round(mean_absolute_percentage_error(y_test, p) * 100, 2),
    }
    fitted[name] = m
    print("  ", results[name])

best = min(results, key=lambda k: results[k]["MAE_usd"])
print(f"\nBest model: {best}")
joblib.dump(fitted[best], f"{OUT}/laptop_price_model.joblib", compress=3)

# Feature importance via permutation (on a test subset for speed)
from sklearn.inspection import permutation_importance
sub = X_test.sample(5000, random_state=1)
pi = permutation_importance(fitted[best], sub, y_test.loc[sub.index], n_repeats=3,
                            random_state=0, n_jobs=-1, scoring="neg_mean_absolute_error")
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
imp = (imp / imp.sum() * 100).round(2)
print("\nTop feature importances (%):\n", imp.head(10))

json.dump({"best_model": best, "results": results, "feature_importance_pct": imp.to_dict()},
          open(f"{OUT}/model_metrics.json", "w"), indent=2)

# Plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pred = fitted[best].predict(X_test)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
idx = np.random.default_rng(0).choice(len(y_test), 4000, replace=False)
ax[0].scatter(y_test.values[idx], pred[idx], s=5, alpha=0.4)
lim = [0, max(y_test.max(), pred.max())]
ax[0].plot(lim, lim, "r--"); ax[0].set(title=f"Actual vs Predicted ({best})", xlabel="Actual USD", ylabel="Predicted USD")
ax[1].hist((pred - y_test.values), bins=80, color="teal"); ax[1].set(title="Residuals (USD)", xlabel="Error")
imp.head(12)[::-1].plot.barh(ax=ax[2], color="darkorange"); ax[2].set(title="Feature importance (%)")
plt.tight_layout(); plt.savefig(f"{OUT}/model_report.png", dpi=110)
print("Saved model, metrics, and plot.")
