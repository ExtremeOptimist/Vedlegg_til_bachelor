import cv2, numpy as np

# Innhenting av bilde og forbredende behandling
img = cv2.imread("Github/Eksperimentelt/Bilder/havbunn/cont_orig.jpg")
img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Definasjon av rødfarge, Limits=[Øvre grense , Nedre grense]
limits = [np.array([0, 30, 70], dtype='uint8')
    ,np.array([5, 255, 255], dtype='uint8')]
# Maskering
mask = cv2.inRange(hsv, limits[0], limits[1])

# Finn konturer fra sorthvitt bilde.
contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Konturer, hierarki = cv2.findContours(bilde, konturinnhentingsmodus,tilnærmingsmetode)

# Kontur liste sortert etter areal.
contours.sort(key=cv2.contourArea, reverse=True)

x,y,w,h = cv2.boundingRect(contours[3])


tom = np.zeros_like(img)
tom = cv2.drawContours(img,contours,-1,(255,21,211),1)

cv2.imshow('1',tom)
cv2.waitKey()
cv2.imshow('2',img)
cv2.waitKey()

bilde= img.copy()
bare_en_kontur = contours[0]

# Tegne alle konturene fra en liste.
cv2.drawContours(bilde, contours,-1, (0,255,0), 3)
# Tegne inn kontur med indeks nr 3.
cv2.drawContours(bilde, contours, 3, (0,255,0), 3)
# Tegne inn en enkelt kontur. Denne konturen er fylt
cv2.drawContours(bilde, [bare_en_kontur], 0, (0,255,0), thickness=-1)
# Eksempel der alle parameter er manuelt definert.
cv2.drawContours(image=bilde,contours=bare_en_kontur,contourIdx=0,color=(159,32,12),
                 thickness=4,lineType=cv2.LINE_AA,hierarchy=None,maxLevel=None,offset=(40,40))
'''
cv2.imshow('fdf', mask)
cv2.waitKey()

cv2.drawContours(mask, cont, -1, 255, 3)  # Draw filled contour in mask

area = 0
for c in cnts:
    area += cv2.contourArea(c)
    cv2.drawContours(original,[c], 0, (0,0,0), 2)

    contours,  = cv2.findContours(gray_tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        moment = cv2.moments(my_cont)  # Finding center of countour
        centerx = int(moment['m10'] / (moment['m00'] + 1e-6))
        centery = int(moment['m01'] / (moment['m00'] + 1e-6))
        img = cv2.drawContours(img, my_cont, 1, (255, 0, 0), 3)
'''