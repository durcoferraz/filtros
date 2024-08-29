import cv2
import numpy as np
import matplotlib.pyplot as plt

rgb = cv2.imread("C:/Users/joaof/Desktop/MestradoUEM/Pesquisa/07_agosto/menor.jpg")
noir = cv2.imread("C:/Users/joaof/Desktop/MestradoUEM/Pesquisa/07_agosto/ir_menor.png")

if rgb is None or noir is None:
    print("Erro ao carregar as imagens. Verifique os caminhos.")
else:
    def calc_ndvi(img, img2):
        b, g, r = cv2.split(img)
        b2, g2, r2 = cv2.split(img2)
        bottom = r.astype(float) + 3 * r2.astype(float)
        bottom[bottom == 0] = 0.01
        sub = 3 * r2 - r
        ndvi = sub / bottom
        return ndvi

    ndvi2 = calc_ndvi(rgb, noir)

    def contrast_stretch(im):
        in_min = np.percentile(im, 5)
        in_max = np.percentile(im, 95)
        out_min = 0.0
        out_max = 255.0
        out = im - in_min
        out *= ((out_min - out_max) / (in_min - in_max))
        out += in_min
        return out

    ndvi_contrasted = contrast_stretch(ndvi2)
    color_mapped_prep = ndvi_contrasted.astype(np.uint8)
    color_mapped_image = cv2.applyColorMap(color_mapped_prep, cv2.COLORMAP_JET)

    # Use matplotlib para exibir a imagem
    plt.imshow(cv2.cvtColor(color_mapped_image, cv2.COLOR_BGR2RGB))
    plt.show()