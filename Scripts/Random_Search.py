import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from scipy.stats import randint

# 1. Load Dataset
data = pd.read_csv('term_weighting_results.csv')

# Pisahkan fitur dan label
X = data.iloc[:, :-1]  # Semua kolom kecuali kolom terakhir sebagai fitur
y = data.iloc[:, -1]   # Kolom terakhir sebagai label

# 2. Split Data (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Baseline Model - Random Forest default tanpa tuning
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)
base_error = 1 - accuracy_score(y_test, y_pred_base)
print(f"Baseline Error Rate: {base_error:.2%}")

# 4. Definisikan parameter distribusi untuk RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 500),                    # Jumlah pohon antara 50-500
    'max_depth': [None] + list(np.arange(5, 50, 5)),     # None atau kedalaman 5-50 step 5
    'min_samples_split': randint(2, 20),                 # Minimum sampel untuk split node
    'min_samples_leaf': randint(1, 10),                  # Minimum sampel di leaf node
    'max_features': ['sqrt', 'log2', None],              # Jumlah fitur untuk split
    'bootstrap': [True, False],                           # Menggunakan bootstrap sampling atau tidak
    'criterion': ['gini', 'entropy'],                     # Kriteria split
    'class_weight': [None, 'balanced', 'balanced_subsample']  # Menangani imbalance class
}

# 5. Inisialisasi model Random Forest
rf = RandomForestClassifier(random_state=42)

# 6. Setup RandomizedSearchCV dengan 100 iterasi dan 5-fold CV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# 7. Jalankan Randomized Search pada data training
random_search.fit(X_train, y_train)

# 8. Tampilkan hasil terbaik
print("\n=== Optimization Results ===")
print(f"Baseline Error: {base_error:.2%}")

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
optimized_error = 1 - accuracy_score(y_test, y_pred)

print(f"Optimized Error: {optimized_error:.2%}")
print(f"Improvement: {(base_error - optimized_error):.2%}")

print("\nBest Parameters:")
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")

# 9. Evaluasi performa model terbaik pada data testing
print(f"\nAccuracy on Test Set: {accuracy_score(y_test, y_pred):.2%}")
print(f"AUC Score: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 11. Tampilkan 10 fitur terpenting
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n10 Fitur Terpenting:")
print(feature_importances.head(10))

# 12. Visualisasi Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted-Negatif', 'Predicted-Positif'],
            yticklabels=['Actual-Negatif', 'Actual-Positif'])
plt.title('Confusion Matrix (Optimized Model)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 13. Visualisasi Feature Importance
plt.figure(figsize=(10, 6))
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]  # Urutkan descending
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Tampilkan feature terpenting di atas
plt.show()

# 14. Simpan model terbaik ke file .pkl
joblib.dump(best_model, 'optimized_rf_model.pkl')
print("Model saved as 'optimized_rf_model.pkl'")

# 15. Cross-validation untuk validasi tambahan
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")
