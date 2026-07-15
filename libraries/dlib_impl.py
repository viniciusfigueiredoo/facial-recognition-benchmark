import os
import json
import dlib
import numpy as np

# arquivo com modelo treinado para detececao de pontos facais
predictor = dlib.shape_predictor(r"..\modelos\shape_predictor_68_face_landmarks.dat")
# encontra os pontos facais de cada rosto
face_rec = dlib.face_recognition_model_v1(r"..\modelos\dlib_face_recognition_resnet_model_v1.dat")
# detecta os rostos e sua posição
detector = dlib.get_frontal_face_detector()


PASTA_DADOS = r"C:\Users\Murilo"
os.makedirs(PASTA_DADOS, exist_ok=True)




def gerar_embedding(path_individual, nome, matricula, tipo_retorno=1):

    imagem = dlib.load_rgb_image(path_individual)

    # detecta rostos
    faces = detector(imagem, 1)

    if len(faces) == 0:
        print("Nenhum rosto encontrado.")
        return None

    # pega apenas o primeiro rosto
    face = faces[0]

    # pontos faciais
    pontosrosto = predictor(imagem, face)

    # gera o embedding
    embedding = np.array(
        face_rec.compute_face_descriptor(imagem, pontosrosto)
    )

    dados = {
        "nome": nome,
        "matricula": matricula,
        "embedding": embedding.tolist()
    }


    # cria o json
    if tipo_retorno == 1:

        nomearquivo = (
            f"{nome.lower().strip().replace(' ', '_')}_{matricula}.json"
        )

        caminhosalvar = os.path.join(PASTA_DADOS, nomearquivo)

        with open(caminhosalvar, "w", encoding="utf-8") as f:
            json.dump(dados, f, indent=4, ensure_ascii=False)

        print(f"{nome} registrado com sucesso.")

        return dados
    # retorno de um dict
    elif tipo_retorno == 2:

        return {
            "nome": nome,
            "matricula": matricula,
            "embedding": embedding
        }

    else:
        print("retorno invalido")
        return None




def comparar_embedding(path_turma, pasta_JSON):

    lista_jsons = {}

    arquivos = [
        a for a in os.listdir(pasta_JSON)
        if a.endswith(".json")
    ]

    if len(arquivos) == 0:
        print("Nenhum JSON encontrado.")
        return None

    for arquivo in arquivos:

        caminho_arquivo = os.path.join(pasta_JSON, arquivo)

        with open(caminho_arquivo, "r", encoding="utf-8") as f:

            dados = json.load(f)

            lista_jsons[dados["matricula"]] = {
                "nome": dados["nome"],
                "embedding": np.array(dados["embedding"])
            }

    embeddings_turma = []

    imagem_turma = dlib.load_rgb_image(path_turma)

    faces = detector(imagem_turma, 1)

    if len(faces) == 0:
        print("Nenhum rosto encontrado na turma.")
        return None

    for face in faces:

        pontosrosto = predictor(imagem_turma, face)

        embedding = np.array(
            face_rec.compute_face_descriptor(
                imagem_turma,
                pontosrosto
            )
        )

        embeddings_turma.append(embedding)

    rostos_encontrados = []

    total_jsons = len(lista_jsons)

    for matricula in lista_jsons:

        embedding_salvo = lista_jsons[matricula]["embedding"]


        for embedding_turma in embeddings_turma:

            distancia = np.linalg.norm(
                embedding_salvo - embedding_turma
            )

            if distancia < 0.6:


                rostos_encontrados.append({
                    "nome": lista_jsons[matricula]["nome"],
                    "matricula": matricula,
                    "distancia": float(distancia)
                })

                break


    acuracia = (
        len(rostos_encontrados) / total_jsons
    ) * 100

    resultado = {
        "rostos encontrados": rostos_encontrados,
        "acurácia": round(acuracia, 2)
    }

    return resultado

