import subprocess
import sys
import json
import warnings
import os
from pathlib import Path
from utils.worker import MARCADOR



warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Configuração inicial

ROOT_DIR = Path(__file__).resolve().parent

# Alunos cadastrados no benchmark: cada entrada e' a UNICA fonte de verdade
# ligando foto -> nome -> matricula, pra nao dessincronizar como aconteceu antes
alunos_cadastrados = [
    {"foto": ROOT_DIR / "data" / "individual" / "ind_caua.jpeg", "nome": "caua", "matricula": "202411250036"},
]

ft_grupo = ROOT_DIR / "data" / "turma" / "centro.jpeg"
pasta_embeddings = ROOT_DIR / "data" / "JSON"

libraries = {
    "insight_face": "libraries.insight_face",
    "face_recognition": "libraries.face_recognition_lib",
    "dlib": "libraries.dlib_impl",
}


# Execução

def fazer_benchmark(nome_lib: str, modulo: str) -> dict:
    """
    Executa o pipeline completo de uma biblioteca e retorna as métricas
    """

    # config inicial para o worker
    cfg = {
        "nome_lib": nome_lib,
        "modulo": modulo,
        "alunos": [
            {"foto": str(aluno["foto"]), "nome": aluno["nome"], "matricula": aluno["matricula"]}
            for aluno in alunos_cadastrados
        ],
        "ft_grupo": str(ft_grupo),
        "pasta_embeddings": str(pasta_embeddings),
    }

    proc = subprocess.run(
        [sys.executable, "-m", "utils.worker", json.dumps(cfg)], cwd=ROOT_DIR, capture_output=True, text=True
    )

    for linha in proc.stdout.splitlines():
        if linha.startswith(MARCADOR):
            return json.loads(linha[len(MARCADOR):])

    raise RuntimeError(
        f"{nome_lib} falhou (exit {proc.returncode}) :\n{proc.stderr[-2000:]}"
    )


# Exibicão
def print_results(resultados: list[dict]):
    cabecalho = (
        f"{'Biblioteca':<18} {'Tempo (s)':>10} {'Modelos (s)':>12} "
        f"{'CPU méd. (%)':>13} {'Mem. mod. (MB)':>15} {'Mem. pico (MB)':>15} "
        f"{'Rostos':>7} {'Acurácia':>9}"
    )
    separador = "-" * len(cabecalho)

    print("\n" + separador)
    print(cabecalho)
    print(separador)
    for r in resultados:
        print(
            f"{r['biblioteca']:<18} "
            f"{r['tempo_s']:>10} "
            f"{r['tempo_modelos_s']:>12} "
            f"{r['cpu_media_%']:>13} "
            f"{r['memoria_modelos_mb']:>15} "
            f"{r['memoria_pico_mb']:>15} "
            f"{str(r['rostos_encontrados']):>7} "
            f"{str(r['acuracia']):>9}"
        )
    print(separador + "\n")

if __name__ == "__main__":
    os.makedirs(pasta_embeddings, exist_ok=True)

    resultados = []
    for nome_lib, modulo in libraries.items():
        print(f"[benchmark] rodando: {nome_lib}...")
        metricas = fazer_benchmark(nome_lib, modulo)
        resultados.append(metricas)

    print_results(resultados)