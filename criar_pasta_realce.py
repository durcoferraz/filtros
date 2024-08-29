import cv2
import numpy as np
import os

def process_image(image_path, output_dir):
    # Passo 1: Carregar a imagem colorida
    imagem_original = cv2.imread(image_path)

    # Verificar se a imagem foi carregada corretamente
    if imagem_original is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return

    # Passo 2: Separar as bandas de cor (B, G, R)
    b, g, r = cv2.split(imagem_original)

    # Definir o tamanho da máscara de média
    tamanho_mascara = (15, 15)

    # Passo 3: Suavizar cada banda individualmente usando uma máscara de média 11x11
    b_suavizada = cv2.blur(b, tamanho_mascara)
    g_suavizada = cv2.blur(g, tamanho_mascara)
    r_suavizada = cv2.blur(r, tamanho_mascara)

    # Combinar as bandas suavizadas para formar a imagem colorida suavizada
    imagem_suavizada = cv2.merge([b_suavizada, g_suavizada, r_suavizada])

    # Passo 4: Calcular o incremento (diferença entre a imagem original e a suavizada)
    # Para evitar problemas com subtração em uint8, convertemos as imagens para int16
    imagem_original_int = imagem_original.astype(np.int16)
    imagem_suavizada_int = imagem_suavizada.astype(np.int16)

    incremento = imagem_original_int - imagem_suavizada_int

    # Passo 5: Adicionar duas vezes o incremento à imagem original para realçar
    imagem_realce_int = imagem_original_int + 8 * incremento

    # Passo 6: Garantir que os valores dos pixels estejam no intervalo [0, 255]
    # Utilizamos np.clip para limitar os valores e convertemos de volta para uint8
    imagem_realce_clamp = np.clip(imagem_realce_int, 0, 255).astype(np.uint8)

    # Salvar a imagem final com realce
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, imagem_realce_clamp)

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
output_dir = "C:/Users/joaof/Desktop/MestradoUEM/Materias/RDP/PlantDoc_highboost_v15x15/valid/images"

# Processar todas as imagens no diretório de entrada
process_images_in_directory(input_dir, output_dir)
