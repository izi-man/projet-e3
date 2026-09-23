from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# -------------------------
# 1. Création du dataset
# -------------------------

data = {
    "temps_reponse": [
        2.1, 1.8, 3.5, 4.2, 1.2,
        2.8, 5.1, 1.5, 3.1, 4.8,
        2.0, 1.1, 3.8, 4.5, 1.7,
        2.5, 5.5, 1.3, 3.3, 4.1
    ],
    "nb_erreurs": [
        0, 0, 1, 2, 0,
        1, 3, 0, 1, 2,
        0, 0, 2, 2, 0,
        1, 4, 0, 1, 2
    ],
    "score": [
        90, 95, 65, 45, 100,
        75, 30, 92, 70, 50,
        88, 100, 60, 55, 97,
        78, 20, 96, 72, 48
    ],
    "satisfait": [
        1, 1, 0, 0, 1,
        1, 0, 1, 1, 0,
        1, 1, 0, 0, 1,
        1, 0, 1, 1, 0
    ]
}

df = pd.DataFrame(data)


# -------------------------
# 2. Séparation X / y
# -------------------------

X = df[
    [
        "temps_reponse",
        "nb_erreurs",
        "score"
    ]
]

y = df["satisfait"]


# -------------------------
# 3. Train / Test
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# 4. Création du modèle
# -------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# -------------------------
# 5. Entraînement
# -------------------------

model.fit(X_train, y_train)


# -------------------------
# 6. Évaluation
# -------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy : {accuracy:.2f}")
print()
print(classification_report(y_test, predictions))


# -------------------------
# 7. Sauvegarde
# -------------------------

Path("model").mkdir(exist_ok=True)

joblib.dump(
    model,
    "model/random_forest.joblib"
)

print("Modèle sauvegardé dans model/random_forest.joblib")