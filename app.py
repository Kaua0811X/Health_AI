from flask import Flask, request, render_template, jsonify
import joblib
import json
import os
import numpy as np

app = Flask(__name__)

# Carrega modelo e vetor
modelo_doenca = joblib.load("modelo/modelo_doenca.pkl")
vetor = joblib.load("modelo/vetor.pkl")

# Carrega dicionário de especialistas
with open("modelo/especialistas.json", "r", encoding="utf-8") as f:
    especialistas_dict = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    sintomas = ""

    if request.method == "POST":
        sintomas = request.form["sintomas"]
        correcao = request.form.get("correcao")

        sintomas_tratado = sintomas.lower().strip()
        sintomas_vector = vetor.transform([sintomas_tratado])

        # Predição
        doenca_pt = modelo_doenca.predict(sintomas_vector)[0]
        especialista = especialistas_dict.get(doenca_pt, "Clínico Geral")

        # Se houver correção do usuário
        if correcao and correcao.strip():
            correcao = correcao.strip()

            # Armazena no JSON
            novo_registro = {
                "sintomas": sintomas,
                "correcao": correcao
            }

            caminho_corrigido = "dados/correcoes_usuario.json"
            if os.path.exists(caminho_corrigido):
                with open(caminho_corrigido, "r", encoding="utf-8") as f:
                    dados_existentes = json.load(f)
            else:
                dados_existentes = []

            dados_existentes.append(novo_registro)
            with open(caminho_corrigido, "w", encoding="utf-8") as f:
                json.dump(dados_existentes, f, ensure_ascii=False, indent=2)

            # Aprendizado incremental
            try:
                modelo_doenca.partial_fit(sintomas_vector, [correcao], classes=np.unique(modelo_doenca.classes_))
                joblib.dump(modelo_doenca, "modelo/modelo_doenca.pkl")
            except Exception as e:
                print("Erro ao aprender com correção:", e)

            doenca_pt = correcao
            especialista = especialistas_dict.get(doenca_pt, "Clínico Geral")

        resultado = {
            "sintomas": sintomas,
            "doenca_pt": doenca_pt,
            "especialista": especialista
        }

    return render_template("index.html", resultado=resultado, **(resultado or {}))


@app.route("/retrain", methods=["POST"])
def retrain():
    secret = request.args.get("secret")
    if secret != "suachavesecreta123":
        return jsonify({"status": "erro", "mensagem": "Acesso negado"}), 403

    try:
        result = subprocess.run(["python", "treinar_modelo.py"], capture_output=True, text=True)
        if result.returncode == 0:
            return jsonify({"status": "ok", "mensagem": "Modelo re-treinado com sucesso!"})
        else:
            return jsonify({
                "status": "erro",
                "mensagem": "Erro ao treinar o modelo",
                "detalhes": result.stderr
            }), 500
    except Exception as e:
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
