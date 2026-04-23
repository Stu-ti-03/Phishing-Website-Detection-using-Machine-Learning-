"""
train_model.py
--------------
Trains a RandomForestClassifier on a HYBRID dataset (synthetic + optional real CSV)
and saves the model to model/model.pkl.

Run this ONCE before starting the API:
    python train_model.py

# VIVA EXPLANATION:
# We use a hybrid dataset because synthetic data alone may not capture real-world
# phishing patterns. Merging it with a real CSV improves generalization — the model
# sees both controlled examples and actual phishing URLs, making it more robust.
"""

import os
import pickle
import numpy as np
import pandas as pd                            # NEW FEATURE – for loading real CSV data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE NAMES — must match extract_features() order in utils/features.py
# ──────────────────────────────────────────────────────────────────────────────

# NEW FEATURE: Centralised feature name list used for XAI (feature importance)
FEATURE_NAMES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "has_suspicious_keyword",
    "subdomain_count",
    "has_https",          # NEW FEATURE – HTTPS presence (0/1)
    "url_entropy",        # NEW FEATURE – randomness of URL characters
    "special_char_count", # NEW FEATURE – count of %, ?, =, _ characters
]


# ──────────────────────────────────────────────────────────────────────────────
# NEW FEATURE: Real dataset loader
# ──────────────────────────────────────────────────────────────────────────────

def load_real_dataset(filepath: str):
    """
    Load a real labelled phishing dataset from a CSV file.

    Expected CSV columns:
        - 'url'   : raw URL string
        - 'label' : 1 = Phishing, 0 = Legitimate

    # VIVA EXPLANATION:
    # Real datasets (e.g. from Kaggle, PhishTank, UCI) contain URLs collected
    # in the wild. They include edge cases that synthetic data misses, which
    # reduces overfitting and improves real-world accuracy.

    Returns (X, y) numpy arrays using the same feature extractor as the rest
    of the system so training and inference stay perfectly consistent.
    """
    from utils.features import extract_features   # import here to avoid circular import
    from utils.preprocess import clean_url

    df = pd.read_csv(filepath)

    # Validate expected columns exist
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'url' and 'label' columns.")

    X_real, y_real = [], []
    for _, row in df.iterrows():
        try:
            cleaned = clean_url(str(row["url"]))
            features = extract_features(cleaned)
            X_real.append(features)
            y_real.append(int(row["label"]))
        except Exception:
            continue   # skip malformed rows silently

    print(f"   ✅ Loaded {len(X_real)} real samples from {filepath}")
    return np.array(X_real), np.array(y_real)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic dataset generation  (IMPROVED – now includes 3 new features)
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 2000, seed: int = 42):
    """
    Generate synthetic URL feature vectors.

    Features (9 total – 6 original + 3 new):
        0 – url_length
        1 – num_dots
        2 – num_hyphens
        3 – num_digits
        4 – has_suspicious_keyword  (0 or 1)
        5 – subdomain_count
        6 – has_https               (0 or 1)  ← NEW FEATURE
        7 – url_entropy             (float)   ← NEW FEATURE
        8 – special_char_count      (int)     ← NEW FEATURE

    Label:
        1 → Phishing
        0 → Legitimate

    # VIVA EXPLANATION:
    # has_https: Phishing sites often skip HTTPS to save setup effort.
    # url_entropy: Random, auto-generated phishing URLs have high character entropy.
    # special_char_count: Attackers use %, ?, =, _ to embed fake parameters or obfuscate.
    """
    rng = np.random.default_rng(seed)

    # ── Legitimate URLs (label = 0) ──────────────────────────────────────────
    legit_n = n_samples // 2
    legit = np.column_stack([
        rng.integers(10, 40,  size=legit_n),   # short URLs
        rng.integers(1,  3,   size=legit_n),   # few dots
        rng.integers(0,  1,   size=legit_n),   # rare hyphens
        rng.integers(0,  3,   size=legit_n),   # few digits
        rng.integers(0,  2,   size=legit_n),   # rarely suspicious keywords
        rng.integers(0,  2,   size=legit_n),   # 0-1 subdomain
        rng.integers(0,  2,   size=legit_n),   # NEW: mostly has HTTPS (simulate ~70%)
        rng.uniform(2.5, 3.5, size=legit_n),  # NEW: low entropy (readable domains)
        rng.integers(0,  3,   size=legit_n),   # NEW: few special chars
    ])
    # IMPROVED LOGIC: ~70% of legit URLs use HTTPS
    legit[:, 6] = rng.choice([0, 1], size=legit_n, p=[0.3, 0.7])
    legit_labels = np.zeros(legit_n, dtype=int)

    # ── Phishing URLs (label = 1) ────────────────────────────────────────────
    phish_n = n_samples - legit_n
    phish = np.column_stack([
        rng.integers(60, 150, size=phish_n),   # long URLs
        rng.integers(4,  10,  size=phish_n),   # many dots
        rng.integers(2,  6,   size=phish_n),   # multiple hyphens
        rng.integers(4,  15,  size=phish_n),   # many digits
        rng.integers(0,  2,   size=phish_n),   # often suspicious keywords
        rng.integers(2,  6,   size=phish_n),   # deep subdomains
        rng.integers(0,  2,   size=phish_n),   # NEW: mostly NO HTTPS
        rng.uniform(3.8, 5.0, size=phish_n),  # NEW: high entropy (random-looking)
        rng.integers(3,  10,  size=phish_n),   # NEW: many special chars
    ])
    # Force ~70% of phishing samples to have a suspicious keyword
    phish[:, 4] = rng.choice([0, 1], size=phish_n, p=[0.3, 0.7])
    # Force ~80% of phishing samples to have NO HTTPS
    phish[:, 6] = rng.choice([0, 1], size=phish_n, p=[0.8, 0.2])
    phish_labels = np.ones(phish_n, dtype=int)

    X = np.vstack([legit, phish])
    y = np.concatenate([legit_labels, phish_labels])
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Training entry point
# ──────────────────────────────────────────────────────────────────────────────

def train_and_save(real_dataset_path: str = None):
    """
    Main training function.

    Args:
        real_dataset_path: Optional path to a real CSV dataset.
                           If provided, real data is merged with synthetic data.

    # VIVA EXPLANATION:
    # Hybrid training = synthetic base + real-world data.
    # The synthetic data ensures coverage of known phishing patterns.
    # The real data provides diversity and prevents over-engineering.
    """
    print("🔧 Generating synthetic training data …")
    X, y = generate_synthetic_data()

    # NEW FEATURE: Optionally merge with real dataset
    if real_dataset_path and os.path.exists(real_dataset_path):
        print(f"📂 Loading real dataset from: {real_dataset_path}")
        X_real, y_real = load_real_dataset(real_dataset_path)
        X = np.vstack([X, X_real])
        y = np.concatenate([y, y_real])
        print(f"   ✅ Merged dataset size: {len(y)} samples")
    else:
        print("   ℹ️  No real dataset provided — using synthetic data only.")
        print("      Tip: pass a CSV path to train_and_save('data/phishing.csv')")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🌲 Training RandomForestClassifier …")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("\n📊 Evaluation on hold-out test set:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    # NEW FEATURE: Print feature importances for XAI awareness
    print("\n🔍 Feature Importances (for XAI / viva):")
    for name, importance in zip(FEATURE_NAMES, clf.feature_importances_):
        bar = "█" * int(importance * 50)
        print(f"   {name:<25} {importance:.4f}  {bar}")

    # Persist model
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"\n✅ Model saved to {model_path}")


if __name__ == "__main__":
    # To use a real dataset: train_and_save("data/phishing_urls.csv")
    train_and_save()
