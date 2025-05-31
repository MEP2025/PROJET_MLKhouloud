# train_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. Chargement et nettoyage des données
df = pd.read_excel('Online Retail.xlsx')
df.dropna(subset=['CustomerID', 'InvoiceNo', 'Description', 'Country'], inplace=True)
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# 2. Feature Engineering
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceHour'] = df['InvoiceDate'].dt.hour
df['InvoiceDayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['DescriptionLength'] = df['Description'].str.len()
df['CustomerID'] = df['CustomerID'].astype(str)
df = df[df['Country'] == 'United Kingdom']  # Ex. focus UK

# 3. Sélection des features
features = ['Quantity', 'UnitPrice', 'InvoiceHour', 'InvoiceDayOfWeek', 'DescriptionLength', 'Country']
X = df[features]
y = df['TotalPrice']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Prétraitement
num_features = ['Quantity', 'UnitPrice', 'InvoiceHour', 'InvoiceDayOfWeek', 'DescriptionLength']
cat_features = ['Country']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# 6. Pipeline et modèle
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 7. Hyperparameter tuning (GridSearch optionnel)
param_grid = {
    'regressor__n_estimators': [100],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 8. Évaluation
y_pred = grid_search.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# 9. Feature importance
model = grid_search.best_estimator_.named_steps['regressor']
importances = model.feature_importances_
feature_names = (
    num_features +
    list(grid_search.best_estimator_.named_steps['preprocessor']
         .transformers_[1][1].get_feature_names_out(cat_features))
)

feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importances, y=feat_importances.index)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

# 10. Sauvegarde du modèle
joblib.dump(grid_search.best_estimator_, 'best_model.pkl')
