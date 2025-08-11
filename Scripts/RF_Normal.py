# ==============================
# 📌 Import Library
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

# ==============================
# 📌 1. Load Dataset
# ==============================
data = pd.read_csv('term_weighting_results.csv')

# Fitur (X) & Label (y)
X = data.iloc[:, :-1]  # semua kolom kecuali kolom terakhir
y = data.iloc[:, -1]   # kolom terakhir (label)

# ==============================
# 📌 2. Split Data
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==============================
# 📌 3. Inisialisasi & Training Model
# ==============================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    oob_score=True,       # untuk OOB Error
    random_state=42
)
model.fit(X_train, y_train)

# ==============================
# 📌 4. Prediksi & Evaluasi
# ==============================
y_pred = model.predict(X_test)

# Accuracy & Error Rate
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
oob_error = 1 - model.oob_score_

print(f"Accuracy: {accuracy:.2%}")
print(f"Error Rate (Test Data): {error_rate:.2%}")
print(f"OOB Error: {oob_error:.2%}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# ==============================
# 📌 5. Error Rate per Class
# ==============================
class_names = y.unique()
for i in range(len(class_names)):
    fp = cm[:, i].sum() - cm[i, i]
    fn = cm[i, :].sum() - cm[i, i]
    total_instances_in_class = y_test.value_counts()[class_names[i]]
    class_error = (fp + fn) / total_instances_in_class
    print(f"Error Rate untuk kelas {class_names[i]}: {class_error:.2%}")

# ==============================
# 📌 6. Fitur Terpenting
# ==============================
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Fitur Terpenting:")
print(feature_importances.head(10))

# ==============================
# 📌 7. Visualisasi Confusion Matrix
# ==============================
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted-Negatif', 'Predicted-Positif'],
    yticklabels=['Actual-Negatif', 'Actual-Positif']
)
plt.title('Confusion Matrix')
plt.show()

# ==============================
# 📌 8. Error Rate vs Jumlah Pohon
# ==============================
errors = []
n_trees_range = range(1, 150, 5)

for n in n_trees_range:
    temp_model = RandomForestClassifier(n_estimators=n, random_state=42)
    temp_model.fit(X_train, y_train)
    y_pred_temp = temp_model.predict(X_test)
    errors.append(1 - accuracy_score(y_test, y_pred_temp))

plt.figure(figsize=(6, 4))
plt.plot(n_trees_range, errors, marker='o')
plt.title("Error Rate vs Jumlah Pohon")
plt.xlabel("Number of Trees")
plt.ylabel("Error Rate")
plt.grid()
plt.show()
