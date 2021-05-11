import cv2, numpy as np
from matplotlib import pyplot as plt

bilde = cv2.imread('Github/vedlegg/bilder/teori/perspective.png')
hoyde, bredde = bilde.shape[:2] #Dimensjoner på inngangbilde
src_points = np.array([[0, 0], [0, hoyde-1], [bredde-1, hoyde-1], [bredde-1, 0]], dtype=np.float32)  # Startpunkt
dst_points = np.array([[20, 20], [20, hoyde-10], [bredde-60, hoyde-50], [bredde-60, 50]], dtype=np.float32) # Sluttpunkt

projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
utgangsbilde = cv2.warpPerspective(bilde, projective_matrix, (bredde,hoyde))



cv2.imshow('Input', bilde)
cv2.imshow('Output', utgangsbilde)
cv2.waitKey()