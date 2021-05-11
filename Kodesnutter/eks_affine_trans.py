import cv2, numpy as np
from matplotlib import pyplot as plt

bilde = cv2.imread('Github/vedlegg/bilder/teori/affine.png')
hoyde, bredde = bilde.shape[:2] #Dimensjoner på inngangbilde
src_points = np.array([[0, 0], [0, hoyde-1], [bredde-1, hoyde-1]], dtype=np.float32)  # Startpunkt
fors = 50   # Forskyver sluttpunktene/destinasjonspnktene et antall piksler ut i bildet
            # slik at ikke det ikke havner i negative pikselkordinater.
dst_points = np.array([[fors, fors], [fors, fors + hoyde - 10], [fors + bredde - 60, fors + hoyde - 50]], dtype=np.float32) # Sluttpunkt

affine_matrix = cv2.getAffineTransform(src_points, dst_points)
utgangsbildet = cv2.warpAffine(bilde, affine_matrix, (bredde+10+fors,hoyde+10+fors))


cv2.imshow('Input', bilde)
cv2.imshow('Output', utgangsbildet)
cv2.waitKey()