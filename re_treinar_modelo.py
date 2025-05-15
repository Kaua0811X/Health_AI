import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Carrega o dataset original
df = pd.read_csv("dados/dataset_balanceado.csv")

# Carrega as correções dos usuários (se existirem)
caminho_corrigido = "dados/correcoes_usuario.csv"
if os.path.exists(caminho_corrigido):
    correcoes = pd.read_csv(caminho_corrigido)
    df = pd.concat([df, correcoes], ignore_index=True)

# Separação de dados
X = df["symptom"]
y = df["disease"]

# Vetorização
vetor = TfidfVectorizer()
X_vetor = vetor.fit_transform(X)

# Treinamento
modelo = MultinomialNB()
modelo.fit(X_vetor, y)

# Salvamento
joblib.dump(modelo, "modelo/modelo_doença.pkl")
joblib.dump(vetor, "modelo/vetor.pkl")

print("Modelo re-treinado e salvo com sucesso.")
