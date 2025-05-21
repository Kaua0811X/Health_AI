import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
import numpy as np

# Carrega dataset principal
json_path = "dados/doencas_sintomas_especialistas_pt.json"
with open(json_path, "r", encoding="utf-8") as f:
    dados = json.load(f)

linhas = []
especialistas_dict = {}

for item in dados:
    doenca = item["doenca"]
    especialista = item.get("especialista", "Clínico Geral")
    especialistas_dict[doenca] = especialista

    for sintoma in item["sintomas"]:
        linhas.append({"symptom": sintoma.lower().strip(), "diagnosis": doenca})

df = pd.DataFrame(linhas)

# Adiciona correções do usuário, se houver
correcao_path = "dados/correcoes_usuario.json"
if os.path.exists(correcao_path):
    with open(correcao_path, "r", encoding="utf-8") as f:
        correcoes = json.load(f)

    correcoes_df = pd.DataFrame(correcoes)
    if "sintomas" in correcoes_df.columns and "correcao" in correcoes_df.columns:
        correcoes_df = correcoes_df[["sintomas", "correcao"]]
        correcoes_df.columns = ["symptom", "diagnosis"]
        correcoes_df["symptom"] = correcoes_df["symptom"].str.lower().str.strip()
        correcoes_df["diagnosis"] = correcoes_df["diagnosis"].str.strip()
        df = pd.concat([df, correcoes_df], ignore_index=True)

# Vetorização e treino com partial_fit
X = df["symptom"]
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

modelo = MultinomialNB()
modelo.partial_fit(X_train_vec, y_train, classes=np.unique(y))

# Avaliação
y_pred = modelo.predict(X_test_vec)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Salva tudo
os.makedirs("modelo", exist_ok=True)
joblib.dump(modelo, "modelo/modelo_doenca.pkl")
joblib.dump(vectorizer, "modelo/vetor.pkl")
with open("modelo/especialistas.json", "w", encoding="utf-8") as f:
    json.dump(especialistas_dict, f, ensure_ascii=False, indent=2)
