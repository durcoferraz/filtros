
import cv2

# Munif

rgb = cv2.imread('C:/Users/Manna/Desktop/teste/programa/rgb.jpg')
noir = cv2.imread('C:/Users/Manna/Desktop/teste/programa/noir.jpg')

def calc_ndvi(img,img2):
  b, g, r = cv2.split(img)
  b2, g2, r2 = cv2.split(img2)
  bottom = r.astype(float) + r2.astype(float)
  bottom[bottom==0] = 0.01
  #bottom[bottom>255] = 255
  sub = ( r2 - r )
  #sub[sub<0] = 0
  ndvi = sub / bottom
  return ndvi

ndvi2 = calc_ndvi(rgb, noir)

cv2.imshow('rgb',rgb)
cv2.imshow('ndvi',ndvi2)

cv2.waitKey(0)
cv2.destroyAllWindows()