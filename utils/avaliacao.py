from pathlib import Path

""" 
Gabarito manual para checagem das respostas do benchmark. O gabarito é um dicionário com a seguinte estrutura:
    { "nome_da_foto": {"matricula_aluno_presente_na_foto"} }
"""
GABARITO = {
    "centro.jpeg": {"202411250036", "202411250037"}, # ambos presentes na foto
    "ala_direita.jpeg": {"202411250037"}, # apenas vitor
    "ala_esquerda.jpeg": {"202411250036"}, # apenas caua
    "ampla.jpeg": {"202411250036", "202411250037"} # ambos presentes na foto
}


def taxas(vp, fp, fn, vn) -> dict:
    """Calcula acuracia/precisao/recall a partir de contagens ja somadas."""
    total = vp + fp + fn + vn
    return {
        "vp": vp, "fp": fp, "fn": fn, "vn": vn,
        "acuracia": round((vp + vn) / total * 100, 2) if total else None,
        "precisao": round(vp / (vp + fp) * 100, 2) if (vp + fp) else None,
        "recall":   round(vp / (vp + fn) * 100, 2) if (vp + fn) else None,
    }


def avaliar(caminho_foto, matriculas_reconhecidas, alunos_cadastrados) -> dict:
    """
    Compara se o valor resultando da lib esta batendo com o gabarito
    """

    nome_foto = Path(caminho_foto).name

    # verificar se tem gabarito para a foto utilizada 
    if nome_foto not in GABARITO:
        raise KeyError(
            f"{nome_foto} nao tem registro no gabarito"
        )
    
    presentes = GABARITO.get(nome_foto, set())
    reconhecidas = set(matriculas_reconhecidas)
    universo = {a["matricula"]
                for a in alunos_cadastrados
                }

    # vp = verdadeiro positivo, vn = verdadeiro negativo, fn = falso negativo e fp = falso positivo
    vp = vn = fn = fp = 0
    falsos_neg, falsos_pos = [], []

    for matricula in universo:
        esta_na_foto = matricula in presentes
        foi_reconhecida = matricula in reconhecidas



        if esta_na_foto and foi_reconhecida:
            vp+= 1
        elif esta_na_foto and not foi_reconhecida:
            fn+= 1
            falsos_neg.append(matricula)
        elif not esta_na_foto and foi_reconhecida:
            fp+= 1
            falsos_pos.append(matricula)
        else:
            vn += 1

    # reusa a mesma formula da agregacao, so acrescenta as listas de erros da foto
    resultado = taxas(vp, fp, fn, vn)
    resultado["falsos_positivos"] = falsos_pos
    resultado["falsos_negativos"] = falsos_neg
    return resultado



