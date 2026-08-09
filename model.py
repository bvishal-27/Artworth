import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================
# 1. Load Dataset
# ==============================
print("Loading dataset...")
df = pd.read_csv("ArtWorthAI_Enhanced_Dataset.csv")
print("Original shape:", df.shape)

# ==============================
# 2. Feature Selection
# ==============================
# Drop unhelpful columns
drop_cols = ["Artwork_Name", "Price"]  # Artwork_Name too unique, Price = target
X = df.drop(columns=drop_cols)
y = df["Price"]

print("Selected columns shape:", X.shape)

# ==============================
# 3. Define Features
# ==============================
numeric_features = [
    "Artist Reputation", "Height", "Width", "Weight",
    "Creation_Year", "Artwork_Age", "Previous_Auction_Price"
]

categorical_features = [
    "Sculpture_Type", "Material", "Artist Name",
    "Artist_Alive", "Aesthetic_Descriptor", "Provenance"
]

print("Numeric cols:", numeric_features)
print("Categorical cols:", categorical_features)

# ==============================
# 4. Outlier Removal
# ==============================
print("Removing outliers...")
z_scores = np.abs(stats.zscore(df[numeric_features].fillna(0)))
filtered_entries = (z_scores < 4).all(axis=1)
X, y = X[filtered_entries], y[filtered_entries]
print(f"After outlier removal: {X.shape}")

# ==============================
# 5. Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. Preprocessor
# ==============================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ==============================
# 7. Models to Compare
# ==============================
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

param_grid = {
    "RandomForest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5],
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
    },
}

# ==============================
# 8. Training and Model Selection
# ==============================
best_model = None
best_score = -np.inf
results = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipe,
        param_grid[name],
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {"r2": r2, "mae": mae, "best_params": grid.best_params_}

    print(f"{name} -> R2: {r2:.4f}, MAE: {mae:.2f}")
    print("Best params:", grid.best_params_)

    if r2 > best_score:
        best_score = r2
        best_model = grid.best_estimator_

# ==============================
# 9. Save Best Model
# ==============================
print("\nBest model selected:", type(best_model.named_steps['model']).__name__)
print("Saving best model as best_model.pkl")
joblib.dump(best_model, "best_model.pkl")
  
# ==============================
# 10. Document Assumptions
# ==============================
print("\nNOTE: This model predicts *relative sculpture values*.")
print("Price values are treated as relative indices (not absolute USD/INR).")
