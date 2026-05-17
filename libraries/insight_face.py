import insightface
from insightface.app import FaceAnalysis
import cv2
import json
import os
import numpy as np

# Bibliotecas desnecessárias para o funcionamento
import sys
import warnings

# Limpando o terminal
warnings.filterwarnings("ignore")

sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

app = FaceAnalysis(name='buffalo_l') 
app.prepare(ctx_id=0, det_size=(640, 640))

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Funções auxiliares
def similaridade_cosseno(embedding1, embedding2):
    return np.dot(embedding1, embedding2)/(np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Função de montagem do embedding usando foto do aluno
def gerar_embedding(path_individual, nome, matricula, tipo_retorno=1):
    imagem_individual = cv2.imread(path_individual)
    rosto = app.get(imagem_individual)
    
    if len(rosto) != 1:
        print("Nenhum ou mais de um rosto numa foto individual.")
    else:
        dados = {
            "nome": nome,
            "matricula": matricula,
            "embedding": rosto[0].normed_embedding.tolist()
        }
        
        if tipo_retorno == 1:
            os.makedirs("data/JSON", exist_ok=True)
            path_json = os.path.join("data", "JSON", f"{matricula}.json")
            with open(path_json, "w") as f:
                json.dump(dados, f)
        else:
            return dados

# Comparando embeddings gerados na foto da turma com o aluno buscado  
def comparar_embedding(path_turma, pasta_JSON):
    # Carregando todos os JSONs e montando o database
    database = []
    arquivos = os.listdir(pasta_JSON)
    for arquivo in arquivos:
        if arquivo.endswith(".json"):
            path_completo = os.path.join(pasta_JSON, arquivo)
        
            with open(path_completo, "r") as f:
                dados = json.load(f)
                database.append({
                    "nome": dados["nome"],
                    "matricula": dados["matricula"],
                    "embedding": np.array(dados["embedding"], dtype=np.float32)
                })

    # Processar dados da turma
    imagem_turma = cv2.imread(path_turma)
    rostos = app.get(imagem_turma)

    # Comparar cada rosto contra o database
    reconhecidos = []
    for rosto in rostos:
        embedding_teste = rosto.normed_embedding
        melhor_match = None
        melhor_pontuacao = -1

        for aluno in database:
            pontuacao = similaridade_cosseno(embedding_teste, aluno["embedding"])
            
            if pontuacao > melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor_match = aluno
        
        if melhor_match and melhor_pontuacao > 0.5:
            reconhecidos.append({
                "nome": melhor_match["nome"],
                "matricula": melhor_match["matricula"],
                "pontuacao": round(melhor_pontuacao, 2)  
            })
        
    return {
            "rostos_encontrados": len(reconhecidos),
            "acuracia": round(float(melhor_pontuacao), 2)
        }
            
gerar_embedding(r"data\individual\Vinicius.jpg", "Vinicius Figueiredo", "202411250033")

print(comparar_embedding(r"data\turma\Vinicius-Caua-Murilo.jpg", r"data\JSON"))