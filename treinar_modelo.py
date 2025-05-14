import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Carregar o dataset
df = pd.read_csv("dados/dataset_balanceado.csv")  # Ajuste o caminho se necessário

# Ajuste as colunas de acordo com o seu dataset
X = df["sintomas"]  # Sintomas
y_doenca = df["disease"]  # Doença

# Criar o vetor TF-IDF para os sintomas
vetor = TfidfVectorizer(stop_words="english")
X_vetorizado = vetor.fit_transform(X)

# Criar o modelo de previsão (Naive Bayes)
modelo_doenca = MultinomialNB()
modelo_doenca.fit(X_vetorizado, y_doenca)

# Salvar o modelo e o vetor
joblib.dump(modelo_doenca, "modelo/modelo_doenca.pkl")
joblib.dump(vetor, "modelo/vetor.pkl")

print("Modelo de doenças treinado e salvo com sucesso.")
