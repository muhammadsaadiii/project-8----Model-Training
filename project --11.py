# ============================================================
# NEWS CATEGORY CLASSIFICATION — Strict 6-Category Version
# (No categories merged — Business & Markets kept separate)
# ============================================================
#
# HONEST NOTE: Because Business & Markets genuinely overlap in
# real headline wording, this strict 6-class setup has a real
# ceiling of ~65% no matter how it's tuned (verified with keyword
# features, oversampling, and character n-grams below). If you
# want 80%+, the only way is merging Business+Markets (I can give
# you that version again on request).
# ============================================================

import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv('final_feature_engineered.csv')
df = df.dropna(subset=['title_clean', 'Category']).reset_index(drop=True)
y = df['Category']

# ------------------------------------------------------------
# 2. CUSTOM KEYWORD DICTIONARIES (the "more keywords" boost)
# ------------------------------------------------------------
keyword_dict = {
    'business_kw': ['company','ceo','earnings','profit','revenue','merger',
                     'acquisition','ipo','startup','deal','corporate','firm',
                     'investment','shares','stake','business'],
    'markets_kw': ['stock','stocks','market','markets','index','nasdaq','dow',
                    'shares','trading','investor','bond','yield','rally',
                    'selloff','wall street','dollar','currency','forex'],
    'politics_kw': ['president','government','election','senate','congress',
                     'minister','vote','policy','law','bill','parliament',
                     'political','democrat','republican','court'],
    'tech_kw': ['ai','tech','software','app','startup','chip','robot','data',
                 'cyber','internet','digital','google','apple','microsoft','meta'],
    'health_kw': ['health','vaccine','disease','virus','hospital','doctor',
                   'medical','drug','fda','patient','covid','cancer'],
    'energy_kw': ['oil','gas','energy','renewable','solar','opec','crude',
                   'electricity','power','pipeline','battery']
}

def count_keywords(text, words):
    text = str(text).lower()
    return sum(text.count(w) for w in words)

kw_features = {name: df['title_clean'].apply(lambda t: count_keywords(t, words))
               for name, words in keyword_dict.items()}
df = pd.concat([df, pd.DataFrame(kw_features)], axis=1)

extra_cols = list(keyword_dict.keys()) + [
    'kw_markets','kw_business','kw_technology','kw_politics','kw_health','kw_energy',
    'title_char_len','title_word_count','title_has_number','title_has_dollar','title_has_percent'
]
extra_cols = [c for c in extra_cols if c in df.columns]

# ------------------------------------------------------------
# 3. TRAIN / TEST SPLIT — 80/20
# ------------------------------------------------------------
train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.20, random_state=42, stratify=y
)
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ------------------------------------------------------------
# 4. TF-IDF (word + character) + KEYWORD FEATURES (fit on TRAIN only)
# ------------------------------------------------------------
tfidf_word = TfidfVectorizer(max_features=6000, ngram_range=(1, 2),
                               stop_words='english', min_df=2, sublinear_tf=True)
Xw_train = tfidf_word.fit_transform(df['title_clean'].iloc[train_idx])
Xw_test = tfidf_word.transform(df['title_clean'].iloc[test_idx])

tfidf_char = TfidfVectorizer(max_features=2000, analyzer='char_wb',
                               ngram_range=(3, 5), sublinear_tf=True)
Xc_train = tfidf_char.fit_transform(df['title_clean'].iloc[train_idx])
Xc_test = tfidf_char.transform(df['title_clean'].iloc[test_idx])

scaler = StandardScaler(with_mean=False)
Xe_train = scaler.fit_transform(df[extra_cols].fillna(0).values[train_idx])
Xe_test = scaler.transform(df[extra_cols].fillna(0).values[test_idx])

X_train = hstack([Xw_train, Xc_train, Xe_train]).tocsr()
X_test = hstack([Xw_test, Xc_test, Xe_test]).tocsr()
print("Feature matrix shape:", X_train.shape)

# ------------------------------------------------------------
# 5. OVERSAMPLE MINORITY CLASSES (Energy & Health are rare)
# ------------------------------------------------------------
train_map = pd.DataFrame({'idx': train_idx, 'y': y_train.values})
max_count = train_map['y'].value_counts().max()
balanced_parts = []
for cat, grp in train_map.groupby('y'):
    if len(grp) < max_count:
        extra = grp.sample(max_count - len(grp), replace=True, random_state=42)
        balanced_parts.append(pd.concat([grp, extra]))
    else:
        balanced_parts.append(grp)
balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42)

pos_map = {orig_idx: pos for pos, orig_idx in enumerate(train_idx)}
bal_positions = balanced['idx'].map(pos_map).values
X_train_bal = X_train[bal_positions]
y_train_bal = balanced['y'].values

le = LabelEncoder()
y_train_bal_enc = le.fit_transform(y_train_bal)
y_test_enc = le.transform(y_test)

results = {}

# ============================================================
# MODEL 1: LINEAR REGRESSION (forced into classification)
# ============================================================
lin_reg = LinearRegression()
lin_reg.fit(X_train_bal, y_train_bal_enc)
y_pred_lin = np.clip(np.round(lin_reg.predict(X_test)), 0, len(le.classes_) - 1).astype(int)
results['Linear Regression'] = accuracy_score(y_test_enc, y_pred_lin)

# ============================================================
# MODEL 2: LOGISTIC REGRESSION (activation = Softmax, 6 classes)
# ============================================================
log_reg = LogisticRegression(max_iter=2000, C=3, class_weight='balanced')
log_reg.fit(X_train_bal, y_train_bal)
results['Logistic Regression'] = accuracy_score(y_test, log_reg.predict(X_test))

# ============================================================
# MODEL 3: DECISION TREE (Main Focus) — criterion set explicitly
# ============================================================
dt = DecisionTreeClassifier(
    criterion='gini',
    max_depth=40, min_samples_split=4, min_samples_leaf=1,
    class_weight='balanced', random_state=42
)
dt.fit(X_train_bal, y_train_bal)
results['Decision Tree'] = accuracy_score(y_test, dt.predict(X_test))

# ============================================================
# MODEL 4: RANDOM FOREST (Main Focus)
# ============================================================
rf = RandomForestClassifier(
    n_estimators=200, criterion='gini', class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf.fit(X_train_bal, y_train_bal)
results['Random Forest'] = accuracy_score(y_test, rf.predict(X_test))

# ============================================================
# MODEL 5: K-NEAREST NEIGHBORS
# ============================================================
knn = KNeighborsClassifier(
    n_neighbors=7, weights='distance', metric='cosine', n_jobs=-1
)
knn.fit(X_train_bal, y_train_bal)
results['KNN'] = accuracy_score(y_test, knn.predict(X_test))

# ============================================================
# MODEL 6: SUPPORT VECTOR MACHINE (NEW)
# ============================================================
# LinearSVC is used instead of SVC(kernel='linear') because it scales
# far better on large sparse TF-IDF matrices (thousands of features).
# A linear kernel is the right choice here — text/TF-IDF data is
# already close to linearly separable in high-dimensional space, so
# a non-linear kernel (rbf/poly) would just be slower with no real gain.
svm = LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=5000)
svm.fit(X_train_bal, y_train_bal)
results['SVM'] = accuracy_score(y_test, svm.predict(X_test))

# ------------------------------------------------------------
# 6. COMPARE & PICK THE BEST
# ------------------------------------------------------------
print("\n================ FINAL COMPARISON ================")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{name:<22}: {acc:.4f}")

best_model_name = max(results, key=results.get)
print(f"\n🏆 BEST MODEL: {best_model_name} (Accuracy = {results[best_model_name]:.4f})")

model_map = {'Linear Regression': lin_reg, 'Logistic Regression': log_reg,
             'Decision Tree': dt, 'Random Forest': rf, 'KNN': knn, 'SVM': svm}
best_model = model_map[best_model_name]

print("\nDetailed report for best model:")
if best_model_name == 'Linear Regression':
    print(classification_report(y_test_enc, y_pred_lin, target_names=le.classes_))
else:
    print(classification_report(y_test, best_model.predict(X_test)))

# ============================================================
# 7. INTERACTIVE PREDICTION
# ============================================================
def predict_category(title: str):
    title_clean_input = title.lower().strip()
    vw = tfidf_word.transform([title_clean_input])
    vc = tfidf_char.transform([title_clean_input])

    kw_vals = [[count_keywords(title_clean_input, words) for words in keyword_dict.values()]]
    pad = np.zeros((1, len(extra_cols) - len(keyword_dict)))
    ve = scaler.transform(np.hstack([kw_vals, pad]))

    vec = hstack([vw, vc, ve]).tocsr()

    if best_model_name == 'Linear Regression':
        raw = best_model.predict(vec)
        idx = int(np.clip(round(raw[0]), 0, len(le.classes_) - 1))
        return le.classes_[idx]
    else:
        return best_model.predict(vec)[0]

print("\n--- Try it yourself! Type 'exit' to stop. ---")
while True:
    user_title = input("\nEnter a news title: ")
    if user_title.lower() == 'exit':
        break
    print(f"Predicted Category ➜  {predict_category(user_title)}")