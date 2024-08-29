import cv2
import numpy as np
import os

def process_image(image_path, output_dir):
    # Carregar a imagem colorida
    imagem = cv2.imread(image_path)
    if imagem is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return

    # Separar a imagem em suas três bandas de cor (B, G, R)
    b, g, r = cv2.split(imagem)

    def process_channel(channel):
        # Equalizar o canal
        channel_equalizado = cv2.equalizeHist(channel)


        # Erodindo o canal e aplicando reconstrução por abertura para limpar partículas pequenas
        elemento_estruturante = np.ones((25, 25), np.uint8)
        channel_erodido = cv2.erode(channel_equalizado, elemento_estruturante)

        reconstrucao = channel_erodido.copy()

        while True:
            reconstrucao_anterior = reconstrucao.copy()
            reconstrucao = cv2.dilate(reconstrucao, elemento_estruturante)
            reconstrucao = cv2.min(reconstrucao, channel_equalizado)
            if np.array_equal(reconstrucao, reconstrucao_anterior):
                break

        # Aplicar filtro Laplaciano para ressaltar os detalhes e contornos
        laplaciano = cv2.Laplacian(channel_equalizado, cv2.CV_64F)
        laplaciano = cv2.convertScaleAbs(laplaciano)

        #adicionar o laplaciano a banda
        imagem_final = channel_equalizado + 0.1*laplaciano
        imagem_final = np.clip(imagem_final, 0, 255)

        return imagem_final

    # Processar cada canal individualmente
    b_processed = process_channel(b)
    g_processed = process_channel(g)
    r_processed = process_channel(r)

    # Juntar as bandas processadas para formar uma imagem colorida
    imagem_final = cv2.merge([b_processed, g_processed, r_processed])

    # Salvar a imagem final colorida
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, imagem_final)
    print(f"Imagem processada e salva: {output_path}")

def process_images_in_directory(input_dir, output_dir):
    # Criar diretório de saída, se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Processar todas as imagens no diretório de entrada
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir)

# Caminhos para os diretórios de entrada e saída
input_dir = "C:/Users/joaof/Desktop/MestradoUEM/Materias/RDP/PlantDoc-Tomatoes.v28i.yolov8/test/images"
output_dir = "C:/Users/joaof/Desktop/MestradoUEM/Materias/RDP/PlantDoc_eq_lap/test/images"

# Processar todas as imagens no diretório de entrada
process_images_in_directory(input_dir, output_dir)

