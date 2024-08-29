import cv2
import numpy as np
import os


def process_image(image_path, output_dir):
    # Carregar a imagem em escala de cinza
    imagem = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return

    # Passo 1: Aplicar filtro Highboost
    imagem_borrada = cv2.GaussianBlur(imagem, (15, 15), 0)

    # Criar a máscara (diferença entre a imagem original e a borrada)
    mascara = cv2.subtract(imagem, imagem_borrada)

    # Aumentar o efeito da máscara
    fator = 8.0  # Ajuste esse valor para intensificar o efeito
    mascara_aumentada = cv2.multiply(mascara, fator)

    # Adicionar a máscara aumentada de volta à imagem original
    highboost = cv2.add(imagem, mascara_aumentada)

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
input_dir = "C:/Users/joaof/Desktop/MestradoUEM/Materias/RDP/PlantDoc-Tomatoes.v28i.yolov8/valid/images"
output_dir = "C:/Users/joaof/Desktop/MestradoUEM/Materias/RDP/PlantDoc_highboost_v15x15a8C/valid/images"

# Processar todas as imagens no diretório de entrada
process_images_in_directory(input_dir, output_dir)
