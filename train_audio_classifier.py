import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

# Paths
DATASET_PATH = "UrbanSound8K/audio/"
CSV_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"

# Feature extraction
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y) < sr:
            raise ValueError("Audio too short")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        return np.hstack([mfccs_mean, mfccs_std])
    except Exception as e:
        print(f"Skipped {file_path}: {e}")
        return None

# Load and process data
def load_data():
    df = pd.read_csv(CSV_PATH)

    # Use actual available classes
    df = df[df['class'].isin(['drilling', 'street_music', 'children_playing'])]

    # Remap class names
    df['class'] = df['class'].replace({
        'drilling': 'sound effect',
        'street_music': 'music',
        'children_playing': 'dialogue'
    })

    print("Loading files...")

    X, y = [], []
    for _, row in df.iterrows():
        label = row['class']
        file_path = os.path.join(DATASET_PATH, f"fold{row['fold']}", row['slice_file_name'])

        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(label)

    print(f"\n✓ Loaded {len(X)} samples")
    return np.array(X), np.array(y)

# Load data
X, y = load_data()

# Show original class distribution
print("\nClass distribution before balancing:")
print(pd.Series(y).value_counts())

# Balance dataset
df = pd.DataFrame(X)
df['label'] = y
min_class_size = df['label'].value_counts().min()

balanced_df = pd.concat([
    df[df['label'] == label].sample(n=min_class_size, random_state=42)
    for label in df['label'].unique()
])

X_bal = balanced_df.drop('label', axis=1).values
y_bal = balanced_df['label'].values

print("\nBalanced class distribution:")
print(pd.Series(y_bal).value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(model, "audio_model.pkl", compress=3)
print("\n✅ Model saved as audio_model.pkl")
