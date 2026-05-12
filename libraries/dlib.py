

def gerar_embedding(path_individual, nome, matricula, tipo_retorno=1):
    '''
	1. Pegar o path e transformar em embedding
	2. Dar a opção de gerar JSON (1) ou não (2)
	se 1, gerar JSON com nome, matricula e embedding
	se 2, return dict(nome="nome", matricula="matricula", embedding="embedding")
    '''
    pass

def comparar_embedding(path_turma, pasta_JSON):
    '''
	1. Gerar embeddings da turma (usando opção 2)
	2. Iterar embeddings nos JSONs com os gerados da turma
	3. return {"rostos encontrados": X, "acurácia": Y}
    '''
    pass