import cv2
import numpy as np
import os

DATASET_PATH = "dataset/faces"

faces = []
labels = []

IMG_SIZE = (100, 100)

for person in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person)

    if os.path.isdir(person_path):

        for img_name in os.listdir(person_path):

            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE
            )

            img = cv2.resize(img, IMG_SIZE)

            faces.append(img.flatten())

            labels.append(person)

faces = np.array(faces)

print("Faces Shape:", faces.shape)
print("Labels:", len(labels))

mean_face = np.mean(faces, axis=0)

print("Mean Face Shape:", mean_face.shape)

A = faces - mean_face

print("Mean Zero Matrix Shape:", A.shape)

C = np.dot(A, A.T)

print("Covariance Shape:", C.shape)

print("Calculating Eigenvalues and Eigenvectors...")

eigenvalues, eigenvectors = np.linalg.eigh(C)

idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues Shape:", eigenvalues.shape)
print("Eigenvectors Shape:", eigenvectors.shape)

k = 100

V = eigenvectors[:, :k]

print("Selected Eigenvectors Shape:", V.shape)

eigenfaces = np.dot(A.T, V)

for i in range(k):
    eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])

print("Eigenfaces Shape:", eigenfaces.shape)

signatures = np.dot(eigenfaces.T, A.T)

print("Signatures Shape:", signatures.shape)

np.save("mean_face.npy", mean_face)
np.save("eigenfaces.npy", eigenfaces)
np.save("signatures.npy", signatures)

print("PCA model saved successfully")

X = signatures.T
y = np.array(labels)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Feature Matrix:", X.shape)
print("Labels:", y.shape)

# ==========================================
# TRAIN TEST SPLIT (60%-40%)
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# KNN TRAINING
# ==========================================

from sklearn.neighbors import KNeighborsClassifier

print("\nTraining KNN Model...")

knn = KNeighborsClassifier(
    n_neighbors=3,
    weights='distance'
)

knn.fit(X_train, y_train)

print("KNN Training Completed")

# ==========================================
# PREDICTION
# ==========================================

predictions = knn.predict(X_test)

# ==========================================
# ACCURACY
# ==========================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# ==========================================
# CONFUSION MATRIX
# ==========================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ==========================================
# SAVE MODEL
# ==========================================

import pickle

with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("\nKNN Model Saved Successfully")
# ==========================================
# IMPOSTER DETECTION
# ==========================================

print("\n==============================")
print("IMPOSTER DETECTION TEST")
print("==============================")

IMPOSTER_PATH = "dataset/imposters"

for img_name in os.listdir(IMPOSTER_PATH):

    img_path = os.path.join(IMPOSTER_PATH, img_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (100, 100))

    test_face = img.flatten()

    # Mean Zero
    test_face = test_face - mean_face

    # Project into Eigenface Space
    test_signature = np.dot(eigenfaces.T, test_face)

    # Standardize
    test_signature = scaler.transform(
        test_signature.reshape(1, -1)
    )

    # Prediction
    prediction = knn.predict(test_signature)[0]

    # Distance from nearest neighbor
    distance = knn.kneighbors(
        test_signature,
        n_neighbors=1
    )[0][0][0]

    THRESHOLD = 12

    if distance > THRESHOLD:

        print(
            img_name,
            " --> Not Enrolled Person",
            "(Distance =", round(distance, 2), ")"
        )

    else:

        print(
            img_name,
            " --> ",
            prediction,
            "(Distance =", round(distance, 2), ")"
        )