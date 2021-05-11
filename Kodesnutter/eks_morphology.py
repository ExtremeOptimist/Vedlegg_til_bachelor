import cv2
# Bilde = Sorthvitt/binært

# Strukturelement:
#  - Vi kan designe egne strukturelement eller hente fra OpenCV
#  - Egenprodusert5x5 helt full med enere.
kernel = np.ones((5,5),np.uint8)
#  - Ferdilaget 5x5 korsform
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))


erosjon = cv2.erode(bilde,kernel,iterations = 1)   # Erosjon
dilasjon = cv2.dilate(bilde,kernel,iterations = 1) # Dilasjon

apning = cv2.morphologyEx(bilde, cv2.MORPH_OPEN, kernel)  # Åpning
lukking = cv2.morphologyEx(bilde, cv2.MORPH_CLOSE, kernel)# Lukking





