import pickle
import numpy as np
from utils.features import extract_features
from utils.preprocess import clean_url

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))

# Test URL (change this anytime)
url = "http://paypal-login-security-check.com/verify-account"

# Preprocess URL
cleaned = clean_url(url)
features = extract_features(cleaned)

# Convert to correct shape (VERY IMPORTANT)
features = np.array(features).reshape(1, -1)

# Prediction
pred = model.predict(features)[0]

# Output
if pred == 1:
    print("Prediction: PHISHING ")
else:
    print("Prediction: LEGITIMATE ")