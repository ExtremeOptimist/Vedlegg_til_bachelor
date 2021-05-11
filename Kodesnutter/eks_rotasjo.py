from Eksperimentelt.klasse_havbunn import rotation, show
import cv2
import numpy as np

bilde = cv2.imread('Github/vedlegg/bilder/teori/affine.png')

resultat = rotation(bilde,-17)

show(resultat)
show(bilde)
#show(np.hstack([bilde,resultat]))