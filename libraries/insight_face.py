from insightface.app import FaceAnalysis
from pathlib import Path
import cv2
import json
import os
import numpy as np

app = FaceAnalysis(name='buffalo_sc')
# det_size padrao do buffalo_sc/det_500m; (1280,1280) faz o detector nao achar
# rostos em fotos individuais (a face fica fora da faixa de escala dos anchors)
app.prepare(ctx_id=0, det_size=(640, 640))

# raiz do projeto, independente do cwd de onde o script foi chamado
ROOT_DIR = Path(__file__).resolve().parent.parent

# subpasta propria em data/JSON, pra nao colidir com embeddings de outras bibliotecas
NOME_BIBLIOTECA = "insight_face"


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
            pasta_lib = ROOT_DIR / "data" / "JSON" / NOME_BIBLIOTECA
            os.makedirs(pasta_lib, exist_ok=True)
            path_json = os.path.join(pasta_lib, f"{matricula}.json")
            with open(path_json, "w") as f:
                json.dump(dados, f)
        else:
            return dados

# Comparando embeddings gerados na foto da turma com o aluno buscado
def comparar_embedding(path_turma, pasta_JSON):
    # Carregando todos os JSONs e montando o database
    pasta_lib = os.path.join(pasta_JSON, NOME_BIBLIOTECA)
    os.makedirs(pasta_lib, exist_ok=True)

    database = []
    arquivos = os.listdir(pasta_lib)
    for arquivo in arquivos:
        if arquivo.endswith(".json"):
            path_completo = os.path.join(pasta_lib, arquivo)
        
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
        
        # buffalo_sc (modelo compacto) gera similaridades mais baixas p/ matches
        # reais; limiar de cosseno ajustado para 0.3 em vez de 0.5
        # alteração feita em 2024-06-10, após testes com fotos de alunos reais
        if melhor_match and melhor_pontuacao > 0.3:
            reconhecidos.append({
                "nome": melhor_match["nome"],
                "matricula": melhor_match["matricula"],
                "pontuacao": round(melhor_pontuacao, 2)  
            })
        
    return {
        "rostos_encontrados": len(rostos),  # total detectado na foto
        "acuracia": round(float(reconhecidos[0]["pontuacao"]), 2) if reconhecidos else 0.0
    }