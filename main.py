from libraries import face_recognition_lib

face_recognition_lib.gerar_embedding("data/individual/caua_teste.jpeg", "caua", "202411250036")
face_recognition_lib.gerar_embedding("data/individual/murilo_teste.jpeg", "murilo", "2024")
print(face_recognition_lib.gerar_embedding("data/individual/murilo_teste.jpeg", "murilo", "202412314"))

print(face_recognition_lib.comparar_embedding("data/turma/trio.jpeg"))