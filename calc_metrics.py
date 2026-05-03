import joblib
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/star_classification.csv')
X = df.drop(columns=['obj_ID', 'run_ID', 'rerun_ID', 'field_ID', 'spec_obj_ID', 'fiber_ID', 'class'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = joblib.load('backend/models/scaler_sdss.pkl')
X_test_scaled = scaler.transform(X_test)

y_test_enc = y_test.map({'GALAXY': 0, 'QSO': 1, 'STAR': 2})

models = {
    'Decision Tree': 'backend/models/model_dt.pkl',
    'Random Forest': 'backend/models/model_rf.pkl',
    'Logistic Regression': 'backend/models/model_logreg.pkl'
}

metrics = {}
for name, path in models.items():
    model = joblib.load(path)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average='weighted')
    metrics[name] = {'accuracy': acc, 'f1_score': f1}

with open('backend/models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print('Metrics saved!')
