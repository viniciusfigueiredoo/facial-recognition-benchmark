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
        registro = []

        if os.path.exists(caminho_arquivo):
            with open(caminho_arquivo, "r", encoding="utf-8") as arq_json:
                conteudo = json.load(arq_json)

            # transforma o conteudo em json ja existente em lista para tratamento
            if isinstance(conteudo, list):
                registro = conteudo
            elif isinstance(conteudo, dict):
                registro = [conteudo]

        # verificar pela matricula a existencia do aluno
        for aluno in registro:
            if aluno["matricula"] == matricula:
                return "Aluno ja cadastrado"

        # salva o registro passado
        registro.append(resultado)

        # salva o arquivo em json
        with open(caminho_arquivo, "w", encoding="utf-8") as arq_json:
            json.dump(registro, arq_json, ensure_ascii=False, indent=4)

    return resultado

def comparar_embedding(path_turma, pasta_JSON = 0):
    '''
	1. Gerar embeddings da turma
	2. Iterar embeddings nos JSONs com os gerados da turma
	3. return {"rostos encontrados": X, "acurácia": Y}
    '''

    list_turma = []
    cont = 0
    img_turma = fc.load_image_file(path_turma)
    encodings_turma = fc.face_encodings(img_turma)

    for encoding in encodings_turma:
        list_turma.append(encoding.tolist())

    with open("data/json_individual/encodings.json", "r", encoding="utf-8") as arq_json:
        conteudo = json.load(arq_json)







    return cont




