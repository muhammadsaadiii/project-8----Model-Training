import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------------------------------
# 1. LOAD THE FEATURE-ENGINEERED FILE
# ------------------------------------------------------------------
df = pd.read_csv("Final_engineered_articles.csv")

# Target: the already-encoded numeric label (created during feature engineering)
y = df["category_label"]

# Features: drop identifier / raw-text columns that aren't numeric features.
# Everything else (tfidf_*, domain_*, sentiment_*, readability_*, kw_*,
# title stats, date features, etc.) is already numeric and ready for the model.
non_feature_cols = ["URL", "Title", "title_clean", "Category", "category_label"]
X = df.drop(columns=[c for c in non_feature_cols if c in df.columns])

# ------------------------------------------------------------------
# 2. TRAIN/TEST SPLIT — test set kept aside, untouched until the end
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 3. HYPERPARAMETER GRIDS
# ------------------------------------------------------------------
dt_param_grid = {
    'max_depth': [5, 10, 15, 20, None]
}

rf_param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'n_estimators': [50, 100, 200, 300]
}

# ------------------------------------------------------------------
# 4. GRID SEARCH + K-FOLD CV (cv=5) FOR EACH MODEL
# ------------------------------------------------------------------
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
dt_grid.fit(X_train, y_train)

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
rf_grid.fit(X_train, y_train)

print("Best Decision Tree params:", dt_grid.best_params_)
print("Decision Tree CV score:   ", dt_grid.best_score_)
print()
print("Best Random Forest params:", rf_grid.best_params_)
print("Random Forest CV score:   ", rf_grid.best_score_)

# ------------------------------------------------------------------
# 5. COMPARE AND PICK THE OVERALL BEST MODEL
# ------------------------------------------------------------------
if rf_grid.best_score_ > dt_grid.best_score_:
    best_model = rf_grid.best_estimator_
    print("\nRandom Forest wins")
else:
    best_model = dt_grid.best_estimator_
    print("\nDecision Tree wins")

# ------------------------------------------------------------------
# 6. FINAL EVALUATION — ONLY NOW TOUCH THE TEST SET
# ------------------------------------------------------------------
y_pred = best_model.predict(X_test)

print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))