import os
import json
from pathlib import Path
import numpy as np
import face_recognition as fc

# raiz do projeto, independente do cwd de onde o script foi chamado
ROOT_DIR = Path(__file__).resolve().parent.parent

# subpasta propria em data/JSON, pra nao colidir com embeddings de outras bibliotecas
NOME_BIBLIOTECA = "face_recognition"


def gerar_embedding(path_individual, nome, matricula=0, tipo_retorno=1):
    '''
	1. Pegar o path e transformar em embedding
	2. Dar a opção de gerar JSON (1) ou não
	se 1, gerar JSON com nome, matricula e embedding
	senão, return dict(nome="nome", matricula="matricula", embedding="embedding")
    '''
    # realiza o load da imagem
    imagem_carregada = fc.load_image_file(path_individual)
    # encontra o encodings de todos rostos presentes na foto
    encodings = fc.face_encodings(imagem_carregada)

    if len(encodings) != 1:
        print("Nenhum ou mais de um rosto numa foto individual.")
    else:
        # formato do dicionario que vai se tornar o JSON
        resultado = {
            "nome": nome,
            "matricula": matricula,
            "embedding": encodings[0].tolist(),
        }

        if tipo_retorno == 1:
            pasta_lib = ROOT_DIR / "data" / "JSON" / NOME_BIBLIOTECA
            os.makedirs(pasta_lib, exist_ok=True)
            path_json = os.path.join(pasta_lib, f"{matricula}.json")
            with open(path_json, "w", encoding="utf-8") as arq_json:
                json.dump(resultado, arq_json, ensure_ascii=False, indent=4)
        else:
            return resultado

def comparar_embedding(path_turma, pasta_JSON):
    '''
	1. Gerar embeddings da turma
	2. Iterar embeddings nos JSONs com os gerados da turma
	3. return {"rostos encontrados": X, "acurácia": Y}
    '''

    # Carregando todos os JSONs e montando o database
    pasta_lib = os.path.join(pasta_JSON, NOME_BIBLIOTECA)
    os.makedirs(pasta_lib, exist_ok=True)

    database = []
    arquivos = os.listdir(pasta_lib)
    for arquivo in arquivos:
        if arquivo.endswith(".json"):
            path_completo = os.path.join(pasta_lib, arquivo)

            with open(path_completo, "r", encoding="utf-8") as arq_json:
                dados = json.load(arq_json)
                database.append({
                    "nome": dados["nome"],
                    "matricula": dados["matricula"],
                    "embedding": np.array(dados["embedding"]),
                })

    # Processar dados da turma
    img_turma = fc.load_image_file(path_turma)
    encodings_turma = fc.face_encodings(img_turma)

    # Comparar cada rosto contra o database
    reconhecidos = []
    for encoding_turma in encodings_turma:
        melhor_match = None
        melhor_pontuacao = -1

        for aluno in database:
            distancia = fc.face_distance([aluno["embedding"]], encoding_turma)[0]
            pontuacao = 1 - distancia

            if pontuacao > melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor_match = aluno

        # limiar padrao do modelo dlib/face_recognition: distancia < 0.6
        # (pontuacao = 1 - distancia, entao pontuacao > 0.4)
        if melhor_match and melhor_pontuacao > 0.4:
            reconhecidos.append({
                "nome": melhor_match["nome"],
                "matricula": melhor_match["matricula"],
                "pontuacao": round(float(melhor_pontuacao), 2)
            })

    return {
        "rostos_encontrados": len(encodings_turma),
        "acuracia": round(float(reconhecidos[0]["pontuacao"]), 2) if reconhecidos else 0.0
    }


