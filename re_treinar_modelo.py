import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from googletrans import Translator
import joblib

# 1. Carregar o dataset original (em inglês)
df = pd.read_csv("dados/dataset_balanceado.csv")

# 2. Traduzir os sintomas e doenças para português
translator = Translator()

def traduzir_lista(textos):
    traduzidos = []
    for texto in textos:
        try:
            t = translator.translate(texto, src='en', dest='pt').text
            traduzidos.append(t)
        except:
            traduzidos.append(texto)  # fallback
    return traduzidos

df["symptom_pt"] = traduzir_lista(df["symptom"])
df["disease_pt"] = traduzir_lista(df["disease"])

# 3. Treinamento
vetor = TfidfVectorizer()
X = vetor.fit_transform(df["symptom_pt"])
y = df["disease_pt"]

modelo = MultinomialNB()
modelo.fit(X, y)

# 4. Salvar os arquivos treinados
joblib.dump(modelo, "modelo_doenca_pt.pkl")
joblib.dump(vetor, "vetor_pt.pkl")

# 5. Salvar CSV traduzido para conferência (opcional)
df.to_csv("dataset_traduzido.csv", index=False)
