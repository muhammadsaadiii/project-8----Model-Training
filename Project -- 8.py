"""
News Category Classifier (Business + Markets merged)
=======================================================
Trains RandomForest, DecisionTree, and LogisticRegression on
final_feature_engineered.csv, picks the best model by test accuracy,
then lets you type in a title and predicts its category.

NOTE: 'Markets' is merged into 'Business' because the two were
semantically overlapping in this dataset (headlines about stocks/
earnings were labeled inconsistently between the two), which was
capping accuracy near 65%. Merging them raises accuracy to ~80%.
"""

import re
import string
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
DATA_PATH = "final_feature_engineered.csv"
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["Category", "Title"]).reset_index(drop=True)

# Merge Markets into Business - the two were heavily overlapping
# and confusing the model (headlines about stocks/earnings/companies
# are ambiguous between the two labels in this dataset)
df["Category"] = df["Category"].replace({"Markets": "Business"})

# Lookup table: title -> actual category (used later for demo predictions)
title_to_category = dict(zip(df["Title"].str.strip().str.lower(), df["Category"]))

# ----------------------------------------------------------------------
# 2. TEXT CLEANING
# ----------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s$%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["title_clean_rebuilt"] = df["Title"].apply(clean_text)

# ----------------------------------------------------------------------
# 3. KEYWORD-COUNT FEATURES (kept from original feature engineering,
#    small extra signal on top of TF-IDF)
# ----------------------------------------------------------------------
keyword_lexicon = {
    "kw_business": ["stock", "stocks", "share", "shares", "nasdaq", "dow", "s&p",
                     "index", "etf", "ipo", "bond", "yield", "trading", "investor",
                     "investors", "market", "markets", "rally", "selloff", "bull",
                     "bear", "ceo", "company", "companies", "revenue", "profit",
                     "earnings", "merger", "acquisition", "startup", "layoffs",
                     "deal", "business", "corporate", "quarterly"],
    "kw_technology": ["ai", "chip", "chips", "software", "app", "tech", "robot",
                        "cyber", "data", "iphone", "google", "microsoft", "meta",
                        "openai", "cloud", "semiconductor", "computing"],
    "kw_politics": ["election", "president", "senate", "congress", "government",
                      "minister", "vote", "policy", "law", "court", "party",
                      "campaign", "biden", "trump", "parliament"],
    "kw_health": ["health", "hospital", "disease", "vaccine", "doctor", "drug",
                    "fda", "medicine", "patient", "virus", "medical", "cancer"],
    "kw_energy": ["oil", "gas", "energy", "solar", "renewable", "opec", "barrel",
                    "electricity", "power", "nuclear", "battery", "crude"],
}

punct_set = set(string.punctuation)

def extract_numeric_features(title):
    t = str(title)
    t_clean = clean_text(t)
    word_count = len(t.split()) if len(t.split()) > 0 else 1

    feats = {
        "title_char_len": len(t),
        "title_word_count": len(t.split()),
        "title_avg_word_len": len(t) / word_count,
        "title_digit_count": sum(c.isdigit() for c in t),
        "title_has_number": int(any(c.isdigit() for c in t)),
        "title_upper_word_count": sum(1 for w in t.split() if w.isupper() and len(w) > 1),
        "title_capitalized_word_count": sum(1 for w in t.split() if w[:1].isupper()),
        "title_has_dollar": int("$" in t),
        "title_has_percent": int("%" in t),
        "title_has_question": int("?" in t),
        "title_has_exclaim": int("!" in t),
        "title_has_colon": int(":" in t),
        "title_punct_count": sum(1 for c in t if c in punct_set),
    }
    feats["title_proper_noun_ratio"] = feats["title_capitalized_word_count"] / word_count

    for feat_name, words in keyword_lexicon.items():
        pattern = r"\b(" + "|".join(words) + r")\b"
        feats[feat_name] = len(re.findall(pattern, t_clean))

    return feats

numeric_feature_names = [
    "title_char_len", "title_word_count", "title_avg_word_len",
    "title_digit_count", "title_has_number", "title_upper_word_count",
    "title_capitalized_word_count", "title_proper_noun_ratio",
    "title_has_dollar", "title_has_percent", "title_has_question",
    "title_has_exclaim", "title_has_colon", "title_punct_count",
    "kw_business", "kw_technology", "kw_politics", "kw_health", "kw_energy",
]

numeric_df = pd.DataFrame([extract_numeric_features(t) for t in df["Title"]])
numeric_df = numeric_df[numeric_feature_names]

# ----------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT (80 / 20, stratified so small classes aren't lost)
# ----------------------------------------------------------------------
X_text = df["title_clean_rebuilt"]
X_numeric = numeric_df
y = df["Category"]

X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text, X_numeric, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------------------
# 5. FEATURE EXTRACTION: word TF-IDF + char TF-IDF + scaled numeric feats
#    (fit ONLY on training data to avoid leakage)
# ----------------------------------------------------------------------
word_tfidf = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
char_tfidf = TfidfVectorizer(max_features=3000, ngram_range=(3, 5), analyzer="char_wb", min_df=2, sublinear_tf=True)

word_train = word_tfidf.fit_transform(X_text_train)
word_test = word_tfidf.transform(X_text_test)
char_train = char_tfidf.fit_transform(X_text_train)
char_test = char_tfidf.transform(X_text_test)

scaler = StandardScaler()
num_train_scaled = scaler.fit_transform(X_num_train)
num_test_scaled = scaler.transform(X_num_test)

X_train = hstack([word_train, char_train, csr_matrix(num_train_scaled)])
X_test = hstack([word_test, char_test, csr_matrix(num_test_scaled)])

# ----------------------------------------------------------------------
# 6. TRAIN 3 MODELS
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", C=8),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=40),
    "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"),
}

results = {}
fitted_models = {}

print("=" * 60)
print("MODEL TRAINING RESULTS")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    fitted_models[name] = model

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

# ----------------------------------------------------------------------
# 7. PICK THE BEST MODEL
# ----------------------------------------------------------------------
best_model_name = max(results, key=results.get)
best_model = fitted_models[best_model_name]

print("=" * 60)
print(f"BEST MODEL: {best_model_name}  (Accuracy: {results[best_model_name]:.4f})")
print("=" * 60)
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:<25}: {acc:.4f}")

# ----------------------------------------------------------------------
# 8. PREDICT ON A USER-ENTERED TITLE
# ----------------------------------------------------------------------
def predict_category(title):
    t_clean = clean_text(title)
    word_vec = word_tfidf.transform([t_clean])
    char_vec = char_tfidf.transform([t_clean])
    num_feats = pd.DataFrame([extract_numeric_features(title)])[numeric_feature_names]
    num_scaled = scaler.transform(num_feats)
    combined = hstack([word_vec, char_vec, csr_matrix(num_scaled)])
    return best_model.predict(combined)[0]

print("\n" + "=" * 60)
print(f"Enter a news title to classify (using {best_model_name})")
print("Type 'exit' to quit")
print("=" * 60)

while True:
    user_title = input("\nEnter title: ").strip()
    if user_title.lower() == "exit":
        break
    if not user_title:
        continue

    predicted = predict_category(user_title)
    actual = title_to_category.get(user_title.lower(), "Not in dataset (new/unseen title)")

    print(f"Predicted Category : {predicted}")
    print(f"Actual Category    : {actual}")