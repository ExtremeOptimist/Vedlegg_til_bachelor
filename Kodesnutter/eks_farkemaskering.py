import cv2, numpy as np
from matplotlib import pyplot as plt
from Eksperimentelt.klasse_havbunn import pshow

# Innlest bilde
img = cv2.imread('Github/Eksperimentelt/Bilder/Konteiner/mate_5.png')

# Formatert fra BGR format til HSV format
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Fargegrenser: Limits = [Øvre grense , Nedre grense]
#       - Grensene angir lilla farger.
limits = [np.array([132, 50, 90], dtype='uint8')
    ,np.array([137, 255, 255], dtype='uint8')]

# Maskering av bildet med angitte fargegrenser
mask = cv2.inRange(hsv, limits[0], limits[1])
    # mask:  sorthvitt bilde:
    #       Hvit == innefor grensene.
    #       Sort == utenfor grensene.
dst= np.zeros_like(img)
# Klipper ut farget omeråde fra original bildet
cut_img = cv2.bitwise_and(img, img, mask=mask)

# Viser bildene

pshow(hsv)
cv2.imshow('Original bildet', img)
cv2.waitKey()
cv2.imshow('Maskering', mask)
cv2.waitKey()
cv2.imshow('Utklipt område', cut_img)
cv2.waitKey()
plt.imshow(cut_img)
plt.show()
plt.imshow(hsv)
plt.show()