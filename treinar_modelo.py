import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Carrega dados principais
df = pd.read_csv("dados/dataset_balanceado.csv")
df.dropna(inplace=True)
df['symptom'] = df['symptom'].str.lower().str.strip()

# Junta com novos dados, se existirem
if os.path.exists("dados/novos_dados.csv"):
    novos = pd.read_csv("dados/novos_dados.csv", names=["symptom", "diagnosis"])
    novos['symptom'] = novos['symptom'].str.lower().str.strip()
    df = pd.concat([df, novos], ignore_index=True)

# Divisão
X = df['symptom']
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)

# Avaliação
y_pred = pipeline.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de classificação:\n", classification_report(y_test, y_pred))
print("Distribuição de classes:\n", df['diagnosis'].value_counts())

# Salva modelo e vetor
joblib.dump(pipeline, "modelo/modelo_doença.pkl")
joblib.dump(pipeline.named_steps["vectorizer"], "modelo/vetor.pkl")
