import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Caminho para o CSV
caminho_csv = "dados/dataset_balanceado.csv"

# Leitura do CSV
df = pd.read_csv(caminho_csv)

# Verificação de colunas e limpeza
df = df[["sintomas", "Disease"]].dropna()

# Separar features e labels
X = df["sintomas"]
y = df["Disease"]

# Dividir entre treino e teste (opcional, mas recomendado para avaliar o modelo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar pipeline com vetorizador e modelo
pipeline = Pipeline([
    ("vetor", TfidfVectorizer()),
    ("modelo", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Avaliação rápida (printa o resultado no console)
y_pred = pipeline.predict(X_test)
print("Relatório de classificação:")
print(classification_report(y_test, y_pred))

# Criar pasta "modelo" se não existir
os.makedirs("modelo", exist_ok=True)

# Salvar o modelo e o vetor
joblib.dump(pipeline.named_steps["modelo"], "modelo/modelo_doença.pkl")
joblib.dump(pipeline.named_steps["vetor"], "modelo/vetor.pkl")

print("✅ Modelo e vetor salvos com sucesso em /modelo")
