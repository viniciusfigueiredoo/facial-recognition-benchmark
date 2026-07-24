"""
Executa o pipeline completo de apenas UMA biblioteca em um processo isolado
"""
#imports de filtro
import warnings
warnings.filterwarnings("ignore")
#imports necessarios para rodar worker
import importlib
import json
import resource
import sys
import time

from pathlib import Path

# sobe pra pasta raiz
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import psutil
from utils.metricas import ColetorMetricas
from utils.avaliacao import avaliar

# constantes usadas pelo codigo
MB = 1024 ** 2
MARCADOR = "###METRICAS###"

def executar(cfg: dict) -> dict:
    processo = psutil.Process()
    rss_base = processo.memory_info().rss

    #sessao onde os imports das libs na main sao medidos a parte
    t0 = time.perf_counter()
    lib = importlib.import_module(cfg["modulo"])
    t_modelos = time.perf_counter() - t0
    rss_modelos = processo.memory_info().rss

    # inicio da contagem
    coletor = ColetorMetricas()
    coletor.comecar()
    t0 = time.perf_counter()

    for aluno in cfg["alunos"]:
        lib.gerar_embedding(aluno["foto"], aluno["nome"], aluno["matricula"])
    resultado = lib.comparar_embedding(cfg["ft_grupo"], cfg["pasta_embeddings"])

    # fim da contagem
    t_pipeline = time.perf_counter() - t0
    coletor.parar()

    # vai pegar o pico de memoria do processo, sem contar seus processos-filhos(RUSAGE_SELF), transformadndo bytes.
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    # impedir que retorne none
    resultado = resultado or {}


    # realizar a verificao com o avaliacao.py

    avaliacao = avaliar(
        cfg["ft_grupo"],
        resultado.get("matriculas_reconhecidas", []),
        cfg["alunos"]
    )


    """
    trabalho total de CPU em nucleo-segundo (cpu% / 100 * tempo). Neutraliza a
    contagem de nucleos: mede quanto de processamento a tarefa custou, seja em
    1 nucleo por muito tempo(dlib e face_recognition) ou em varios nucleos em 
    paralelo por pouco tempo(insight_face).
    """
    trabalho_total = coletor.uso_medio_cpu / 100 * t_pipeline

    return {
        "biblioteca":          cfg["nome_lib"],
        "tempo_s":             round(t_pipeline, 4),
        "tempo_modelos_s":     round(t_modelos, 4),
        "cpu_media_%":         round(coletor.uso_medio_cpu, 2),
        "trabalho_nucleo_s":   round(trabalho_total, 4),
        "memoria_modelos_mb":  round((rss_modelos - rss_base) / MB, 2),
        "memoria_pico_mb":     round((pico - rss_base) / MB, 2),
        "rostos_encontrados":  resultado.get("rostos_encontrados"),
        "avaliacao":           avaliacao,
    }


if __name__ == "__main__":
    print(MARCADOR + json.dumps(executar(json.loads(sys.argv[1]))))