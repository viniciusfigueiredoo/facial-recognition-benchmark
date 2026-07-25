import subprocess
import sys
import json
import warnings
import os
import statistics
from pathlib import Path
from utils.worker import MARCADOR
from utils.avaliacao import GABARITO



warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Configuração inicial

ROOT_DIR = Path(__file__).resolve().parent
REPETICOES = 3

# Alunos cadastrados no benchmark: cada entrada é a UNICA fonte de verdade
# ligando foto -> nome -> matricula, pra nao dessincronizar como aconteceu antes
alunos_cadastrados = [
    {"foto": ROOT_DIR / "data" / "individual" / "ind_caua.jpeg", "nome": "caua", "matricula": "202411250036"},
    {"foto": ROOT_DIR / "data" / "individual" / "ind_vitor.jpeg", "nome": "vitor", "matricula": "202411250037"},


]

fotos_turma = [ROOT_DIR / "data"/ "turma"/ nome for nome in GABARITO]
pasta_embeddings = ROOT_DIR / "data" / "JSON"

libraries = {
    "insight_face": "libraries.insight_face",
    "face_recognition": "libraries.face_recognition_lib",
    "dlib": "libraries.dlib_impl",
}


# Execução

def _extrair_resultado(proc, nome_lib):
    """Le a linha marcada do worker e devolve o dict de metricas."""
    for linha in proc.stdout.splitlines():
        if linha.startswith(MARCADOR):
            return json.loads(linha[len(MARCADOR):])
    raise RuntimeError(
        f"{nome_lib} falhou (exit {proc.returncode}) :\n{proc.stderr[-2000:]}"
    )


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
        "fotos_turma": [str(p) for p in fotos_turma],
        "pasta_embeddings": str(pasta_embeddings),
    }


    # roda o worker REPETICOES vezes, cada uma em processo isolado
    execucoes = []
    for _ in range(REPETICOES):
        proc = subprocess.run(
            [sys.executable, "-m", "utils.worker", json.dumps(cfg)],
            cwd=ROOT_DIR, capture_output=True, text=True,
        )
        execucoes.append(_extrair_resultado(proc, nome_lib))

    # desempenho vira media +/- desvio; a acuracia e' deterministica (mesma em
    # toda repeticao), entao basta pegar de uma execucao
    def resumo(chave):
        vals = [e[chave] for e in execucoes]
        desvio = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"media": round(statistics.mean(vals), 4), "desvio": round(desvio, 4)}

    return {
        "biblioteca":        nome_lib,
        "tempo_s":           resumo("tempo_s"),
        "tempo_modelos_s":   resumo("tempo_modelos_s"),
        "cpu_media_%":       resumo("cpu_media_%"),
        "trabalho_nucleo_s": resumo("trabalho_nucleo_s"),
        "memoria_pico_mb":   resumo("memoria_pico_mb"),
        "repeticoes":        REPETICOES,
        "avaliacao":         execucoes[0]["avaliacao"],
    }


# Exibicão
def print_results(resultados: list[dict]):
    def celula(resumo):
        return f"{resumo['media']} ± {resumo['desvio']}"

    n = resultados[0]["repeticoes"] if resultados else 0
    cabecalho = (
        f"{'Biblioteca':<18} {'Tempo (s)':>18} {'Modelos (s)':>18} "
        f"{'CPU méd. (%)':>20} {'Trab. (nucleo·s)':>20} {'Mem. pico (MB)':>20} "
        f"{'Precisão':>9} {'Recall':>8}"
    )
    separador = "-" * len(cabecalho)

    print(f"\n(media +/- desvio de {n} repeticoes)")
    print(separador)
    print(cabecalho)
    print(separador)
    for r in resultados:
        avaliacao = r["avaliacao"]
        precisao = f"{avaliacao['precisao']}%" if avaliacao["precisao"] is not None else "N/A"
        recall = f"{avaliacao['recall']}%" if avaliacao["recall"] is not None else "N/A"
        print(
            f"{r['biblioteca']:<18} "
            f"{celula(r['tempo_s']):>18} "
            f"{celula(r['tempo_modelos_s']):>18} "
            f"{celula(r['cpu_media_%']):>20} "
            f"{celula(r['trabalho_nucleo_s']):>20} "
            f"{celula(r['memoria_pico_mb']):>20} "
            f"{precisao:>9} "
            f"{recall:>8}"
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