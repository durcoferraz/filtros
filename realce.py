import cv2

image_path = 'C:/Users/Manna/Desktop/teste/programa/08072024-1539_1.jpg'
imagem = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

imagemborrada = cv2.medianBlur(imagem, 23)

mask = cv2.subtract(imagem, imagemborrada)

final = cv2.addWeighted(imagem,1, mask,2.5,0)

cv2.imwrite('imagemcontorno.jpg', imagem)
cv2.imwrite('imagemborradacontorno.jpg', imagemborrada)
cv2.imshow('Imagem original', imagem)
cv2.imshow('imagemborrada', imagemborrada)
cv2.imwrite('maskcontorno.jpg', mask)
cv2.imwrite('finalcontorno.jpg', final)
cv2.imshow('mask', mask)
cv2.imshow('final', final)


cv2.waitKey(0)
cv2.destroyAllWindows()