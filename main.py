from libraries import face_recognition_lib


print(face_recognition_lib.gerar_embedding("data/individual/caua_teste.jpeg", "Cauã", matricula="202411250036",tipo_retorno=1))
print(face_recognition_lib.gerar_embedding("data/individual/murilo_teste.jpeg", "Murilo", matricula="202411250039"))
