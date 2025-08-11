import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

# 1. Load Dataset
data = pd.read_csv('term_weighting_results.csv')

# Pisahkan fitur dan label
X = data.iloc[:, :-1]  # Semua kolom kecuali kolom terakhir sebagai fitur
y = data.iloc[:, -1]   # Kolom terakhir sebagai label

# 2. Split Data (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Baseline Model - Random Forest tanpa tuning hyperparameter
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)
initial_error = 1 - accuracy_score(y_test, y_pred_base)
print(f"Initial Error Rate: {initial_error:.2%}")

# 4. Setup Parameter Grid untuk GridSearchCV (Hyperparameter Tuning)
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# 5. Inisialisasi Random Forest
rf = RandomForestClassifier(random_state=42)

# 6. Setup GridSearchCV dengan 5-fold CV dan semua core CPU
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# 7. Jalankan Grid Search pada training data
grid_search.fit(X_train, y_train)

# 8. Tampilkan hasil terbaik dari Grid Search
print("\nParameter terbaik:", grid_search.best_params_)
print("Akurasi terbaik (CV): {:.2f}%".format(grid_search.best_score_ * 100))

# 9. Evaluasi model terbaik pada data test
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
optimized_error = 1 - accuracy_score(y_test, y_pred)

print("\n=== Optimization Results ===")
print(f"Initial Error Rate: {initial_error:.2%}")
print(f"Optimized Error Rate: {optimized_error:.2%}")
print(f"Error Reduction: {(initial_error - optimized_error):.2%}")

print("\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

print("\nAkurasi pada test set: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 10. Tampilkan 10 fitur terpenting
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\n10 Fitur Terpenting:")
print(feature_importances.head(10))

# 11. Visualisasi Confusion Matrix dan Feature Importance
plt.figure(figsize=(12, 6))

# Confusion Matrix
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted-Negatif', 'Predicted-Positif'],
            yticklabels=['Actual-Negatif', 'Actual-Positif'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Feature Importance (Top 10)
plt.subplot(1, 2, 2)
importances = best_rf.feature_importances_
indices = np.argsort(importances)[-10:]
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title('Top 10 Feature Importances')

plt.tight_layout()
plt.show()

# 12. Simpan model terbaik ke file .pkl
joblib.dump(best_rf, 'best_rf_model.pkl')
