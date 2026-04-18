import os
import json
import face_recognition as fc


def gerar_embedding(path_individual, nome, matricula=0, tipo_retorno=1):
    '''
	1. Pegar o path e transformar em embedding
	2. Dar a opção de gerar JSON (1) ou não
	se 1, gerar JSON com nome, matricula e embedding
	senão, return dict(nome="nome", matricula="matricula", embedding="embedding")
    '''
    list_encodings = []


    # realiza o load da imagem
    imagem_carregada = fc.load_image_file(path_individual)
    # encontra o encodings de todos rostos presentes na foto
    encodings = fc.face_encodings(imagem_carregada)

    for encoding in encodings:
        list_encodings.append(encoding.tolist())

    # formato do dicionario que vai se tornar o JSON
    resultado = {
        "nome": nome,
        "matricula": matricula,
        "embedding": list_encodings,
    }

    if tipo_retorno == 1:
        pasta_destino = "data/json_individual"
        caminho_arquivo = os.path.join(pasta_destino, "encodings.json")
        registros = []

        # caso o arquivo encondings.json já exista para não apagar os encodings existentes
        if os.path.exists(caminho_arquivo):
            with open(caminho_arquivo, "r", encoding="utf-8") as arq_json:
                conteudo_existente = json.load(arq_json)

            if isinstance(conteudo_existente, list):
                registros = conteudo_existente
            elif isinstance(conteudo_existente, dict):
                registros = [conteudo_existente]

        registros.append(resultado)

        # escrever o arquivo encondings.json na pasta data/json_individual
        with open(caminho_arquivo, "w", encoding="utf-8") as arq_json:
            json.dump(registros, arq_json, ensure_ascii=False, indent=4)

    return resultado

def comparar_embedding(path_turma, pasta_JSON):
    '''
	1. Gerar embeddings da turma (usando opção 2)
	2. Iterar embeddings nos JSONs com os gerados da turma
	3. return {"rostos encontrados": X, "acurácia": Y}
    '''
    pass


