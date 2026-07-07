import time
import os
from utils.metricas import ColetorMetricas
import libraries.insight_face as insightface
import libraries.face_recognition_lib as face_rec
#import libraries.dlib as dlib

# Configuração inicial

foto_individual = r"data\individual\vinicius.jpg"
foto_grupo = r"data\turma\grupo.jpg"
pasta_embeddings = r"data\JSON"

nome_teste = "Vinicius"
matricula_teste = "202411250033"

libraries = {
    "insight_face": insightface,
    "face_recognition": face_rec,
    #"dlib": dlib,
}


# Execução

def fazer_benchmark(name: str, lib) -> dict:
    """
    Executa o pipeline completo de uma biblioteca e retorna as métricas
    """

    coletor = ColetorMetricas()

    # início da medição
    coletor.comecar()
    t_inicial = time.perf_counter()

    lib.gerar_embedding(foto_individual, nome_teste, matricula_teste)
    resultado = lib.comparar_embedding(foto_grupo, pasta_embeddings)

    t_final = time.perf_counter()
    coletor.parar()
    # fim da medição

    return {
        "biblioteca":          name,
        "tempo_s":             round(t_final - t_inicial, 4),
        "cpu_media_%":         round(coletor.uso_medio_cpu, 2),
        "memoria_pico_mb":     round(coletor.pico_memoria_mb, 2),
        "rostos_encontrados":  resultado.get("rostos_encontrados"),
        "acuracia":            resultado.get("acuracia"),
    }


# Exibição

def print_results(resultados: list[dict]):
    cabecalho = f"{'Biblioteca':<20} {'Tempo (s)':>10} {'CPU méd. (%)':>14} {'Mem. pico (MB)':>16} {'Rostos':>8} {'Acurácia':>10}"
    separador = "-" * len(cabecalho)

    print("\n" + separador)
    print(cabecalho)
    print(separador)
    for r in resultados:
        print(
            f"{r['biblioteca']:<20}"
            f"{r['tempo_s']:>10}"
            f"{r['cpu_media_%']:>14}"
            f"{r['memoria_pico_mb']:>16}"
            f"{str(r['rostos_encontrados']):>8}"
            f"{str(r['acuracia']):>10}"
        )
    print(separador + "\n")


# Ponto de entrada

if __name__ == "__main__":
    os.makedirs(pasta_embeddings, exist_ok=True)

    resultados = []
    for name, lib in libraries.items():
        print(f"[benchmark] rodando: {name}...")
        metricas = fazer_benchmark(name, lib)
        resultados.append(metricas)

    print_results(resultados)