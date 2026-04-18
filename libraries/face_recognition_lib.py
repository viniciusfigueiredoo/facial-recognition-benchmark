import os

import face_recognition as fc

def gerar_embedding(path_individual, nome = "default", matricula = 0, tipo_retorno=1):
    '''
	1. Pegar o path e transformar em embedding
	2. Dar a opção de gerar JSON (1) ou não (2)
	se 1, gerar JSON com nome, matricula e embedding
	se 2, return dict(nome="nome", matricula="matricula", embedding="embedding")
    '''
    list_arquivos = []
    list_encondings = []
    # path_individual é uma pasta
    # for para encontrar os arquivos na pasta individual
    for arquivo in os.listdir(path_individual):
        # path dos arquivos
        arquivos_analisados = os.path.join(path_individual, arquivo)

        # verificar se são arquivos .jpeg
        if os.path.isfile(arquivos_analisados) and arquivos_analisados.lower().endswith(".jpeg"):
            list_arquivos.append(arquivos_analisados)


    for arquivo in list_arquivos:
        # carregou a imagem
        imagem_carregada = fc.load_image_file(arquivo)
        # transformou em encoding
        list_encondings = fc.face_encodings(imagem_carregada)



    return list_arquivos, list_encondings

def comparar_embedding(path_turma, pasta_JSON):
    '''
	1. Gerar embeddings da turma (usando opção 2)
	2. Iterar embeddings nos JSONs com os gerados da turma
	3. return {"rostos encontrados": X, "acurácia": Y}
    '''
