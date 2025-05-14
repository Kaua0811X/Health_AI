from flask import Flask, request, render_template, jsonify
import joblib
import datetime
import json
import os

app = Flask(__name__)
modelo_doenca = joblib.load("modelo/modelo_doenca.pkl")
vetor = joblib.load("modelo/vetor.pkl")

# Mapeamento de doenças para especialidades
mapeamento_especialidade = {
    "Gastroenterite": "Gastroenterologia",
    "Dengue": "Infectologia",
    "Hipertensão": "Cardiologia",
    # Adicione mais doenças e especialidades conforme necessário
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/triagem", methods=["POST"])
def triagem():
    try:
        dados = request.json
        sintomas = dados["sintomas"]
        nome = dados["nome"]
        cpf = dados["cpf"]
        rg = dados["rg"]

        # Vetorizar os sintomas
        sintomas_vet = vetor.transform([sintomas])
        doenca_predita = modelo_doenca.predict(sintomas_vet)[0]

        # Mapear a especialidade
        especialidade = mapeamento_especialidade.get(doenca_predita, "Especialidade não encontrada")

        ficha = {
            "nome": nome,
            "cpf": cpf,
            "rg": rg,
            "sintomas": sintomas,
            "doenca_sugerida": doenca_predita,
            "especialidade_sugerida": especialidade,
            "data_hora": datetime.datetime.now().isoformat()
        }

        # Salvar ficha
        os.makedirs("fichas", exist_ok=True)
        with open(f"fichas/{nome}_{cpf}.json", "w", encoding="utf-8") as f:
            json.dump(ficha, f, ensure_ascii=False, indent=4)

        return jsonify(ficha)
    
    except Exception as e:
        return jsonify({"erro": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
