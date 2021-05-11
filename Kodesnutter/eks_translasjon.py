from Eksperimentelt.klasse_havbunn import translation, show
import cv2
import numpy as np

bilde = cv2.imread('Github/vedlegg/bilder/teori/affine.png')

resultat = translation(bilde,50,20)

show(resultat)
show(bilde)
#show(np.hstack([bilde,resultat]))