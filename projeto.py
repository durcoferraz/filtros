import cv2
import numpy as np
import os


def process_image(image_path, output_dir):
    # Carregar a imagem em escala de cinza
    imagem = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return

    # Passo 1: Remover ruídos usando reconstrução por abertura
    elemento_estruturante = np.ones((21, 21), np.uint8)
    imagem_abertura = cv2.morphologyEx(
        imagem, cv2.MORPH_OPEN, elemento_estruturante)
    marcador = imagem_abertura
    reconstrucao = marcador.copy()

    while True:
        reconstrucao_anterior = reconstrucao.copy()
        reconstrucao = cv2.dilate(reconstrucao, elemento_estruturante)
        reconstrucao = cv2.min(reconstrucao, imagem)
        if np.array_equal(reconstrucao, reconstrucao_anterior):
            break

    # Passo 3: Aplicar filtro mediano com matriz 5x5
    filtro_mediano = cv2.medianBlur(reconstrucao, 11)

    # Passo 4: Usar o resultado do filtro mediano como marcador para uma nova reconstrução
    marcador2 = filtro_mediano
    reconstrucao2 = marcador2.copy()

    while True:
        reconstrucao_anterior2 = reconstrucao2.copy()
        reconstrucao2 = cv2.dilate(reconstrucao2, elemento_estruturante)
        reconstrucao2 = cv2.min(reconstrucao2, reconstrucao)
        if np.array_equal(reconstrucao2, reconstrucao_anterior2):
            break

    # Passo 5: Aplicar filtro Highboost
    imagem_borrada = cv2.GaussianBlur(reconstrucao2, (11, 11), 0)
    highboost = cv2.subtract(reconstrucao2, imagem_borrada)
    highboost = cv2.add(reconstrucao2, highboost)

    # Salvar imagem final (highboost)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, highboost)


def process_images_in_directory(input_dir, output_dir):
    # Criar diretório de saída, se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Processar todas as imagens no diretório de entrada
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir)
            print(f"Imagem processada e salva: {filename}")


# Caminhos para os diretórios de entrada e saída
input_dir = "C:/Users/joaof/Desktop/MestradoUEM/Pesquisa/Testes/teste1/pasta"
output_dir = "C:/Users/joaof/Desktop/MestradoUEM/Pesquisa/Testes/teste1/pastaprocessada"

# Processar todas as imagens no diretório de entrada
process_images_in_directory(input_dir, output_dir)
