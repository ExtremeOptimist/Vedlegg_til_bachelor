import cv2, numpy as np
from matplotlib import pyplot as plt
import time

# Nyttig informasjon:
# img.shape  returnere (rader,kolonner). når vi arbeider med np.array til et bilde er syntaks (rader,colloner) "(y,x)"
# når vi arbeider med points, f eks fra konturer er syntaks (y-kord,xkord)

HSV_colors = {
    'rød': [np.array([0, 50, 80], dtype='uint8'), np.array([5, 255, 255], dtype='uint8')],
    'rød_1': [np.array([2, 50, 50], dtype='uint8'), np.array([7, 150, 150], dtype='uint8')],
    'rød_2': [np.array([177, 100, 100], dtype='uint8'), np.array([179, 255, 255], dtype='uint8')],
    'gul': [np.array([20, 20, 20], dtype='uint8'), np.array([30, 255, 255], dtype='uint8')],
    'grønn': [np.array([121, 20, 0], dtype='uint8'), np.array([180, 255, 255], dtype='uint8')],
    'turkus': [np.array([181, 20, 0], dtype='uint8'), np.array([240, 255, 255], dtype='uint8')],
    'blå': [np.array([105, 30, 30], dtype='uint8'), np.array([115, 255, 255], dtype='uint8')],
    'lilla': [np.array([170, 20, 20], dtype='uint8'), np.array([179, 255, 255], dtype='uint8')],
    'blå_lilla': [np.array([110, 70, 100], dtype='uint8'), np.array([179, 255, 255], dtype='uint8')],
    'issoler_farger': [np.array([0, 70, 0], dtype='uint8'), np.array([179, 255, 255], dtype='uint8')],
    'sort': [np.array([0, 0, 0], dtype='uint8'), np.array([179, 130, 130], dtype='uint8')],
    'svamp': [np.array([12, 30, 0], dtype='uint8'), np.array([20, 170, 150], dtype='uint8')]
}

class Square:
    def __init__(self, img, hsv=None, coordinates=None, pixel_position=None, debug=False):
        self.debug = debug
        self.img = img
        if hsv is None:
            self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            self.hsv = cv2.medianBlur(self.hsv, 5)
        self.hsv = hsv
        self.coordinate = coordinates
        self.pixel_position = pixel_position
        self.funn = None

        # Generelle egenskaper:
        self.area = self.img.shape[0] * self.img.shape[1]

        # Diverse konstanter:
        # OBS justeringer til find_spong gjøres direkt i funksjonen
        self.planting_area_min_solidity = 0.9
        self.planting_area_min_areal = 3000

        self.sea_star_min_areal = 1500
        self.sea_star_max_areal = 3750
        self.sea_star_min_solidity = 0.3
        self.sea_star_max_solidity = 0.6

        self.sponge_min_soliditet = 0.8
        self.sponge_min_areal = 400
        self.sponge_max_areal = 2500
        self.spong_min_fill_of_cicle = 0.66
        self.spong_max_height = 85
        self.spong_min_height = 15
        self.spong_max_width = 60
        self.spong_min_width = 15

        # Alternativ 1 - Full lengde horisontalt rør med T formet endestykke.
        self.coral_reef_alt_1_min_height = 5
        self.coral_reef_alt_1_maks_height = 40
        self.coral_reef_alt_1_min_width = 70
        self.coral_reef_alt_1_maks_width = 91
        self.coral_reef_alt_1_min_extent = 0.15
        self.coral_reef_alt_1_max_extent = 0.75

        # Alternativ 2 - Full lengde enkelt vertikalt rør.
        self.coral_reef_alt_2_min_height = 40
        self.coral_reef_alt_2_maks_height = 91
        self.coral_reef_alt_2_min_width = 8
        self.coral_reef_alt_2_maks_width = 25
        self.coral_reef_alt_2_min_extent = 0.5

        # Alternativ 3 - Hjørnestykke eller avkuttet horisontalt rør med T formet endestykke.
        self.coral_reef_alt_3_min_height = 25
        self.coral_reef_alt_3_maks_height = 80
        self.coral_reef_alt_3_min_width = 25
        self.coral_reef_alt_3_maks_width = 80
        self.coral_reef_alt_3_min_extent = 0.2
        self.coral_reef_alt_3_max_extent = 0.7

        # Alternativ 4 - Stor T-formet fot som ikke har blitt overskygget
        self.coral_reef_alt_4_min_areal = 1000
        self.coral_reef_alt_4_maks_areal = 3000
        self.coral_reef_alt_4_min_width = 40
        self.coral_reef_alt_4_min_height = 40
        self.coral_reef_alt_4_min_extent = 0.2
        self.coral_reef_alt_4_max_extent = 0.4

        # self.scan_for_objects()

    def scan_for_objects(self):
        self.find_designated_planting_area()
        if self.funn is None:
            self.find_sea_star()
        if self.funn is None:
            self.find_sponge()
        if self.funn is None:
            self.find_coral_reef()

    def find_designated_planting_area(self):
        # 1. Sjekk ut gul farge for å finne svamp:
        limits = HSV_colors.get('gul')
        mask = cv2.inRange(self.hsv, limits[0], limits[1])
        if self.debug: show(mask)

        # 2. Henter ut objekter og sortere dem etter areal
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours.sort(key=cv2.contourArea, reverse=True)
        if len(contours) != 0 and cv2.contourArea(contours[0]) > 2:
            # Sortert i synkende rekkefølge etter areal.
            # 3. Sjekker største gule dekker en gitt grense av ruten.
            #     - Hvis dette er tilfellet har vi funnet planteplassen
            areal_convex = cv2.contourArea(cv2.convexHull(contours[0]))
            areal = cv2.contourArea(contours[0])
            solidity = areal / areal_convex
            if solidity > self.planting_area_min_solidity and areal > self.planting_area_min_areal:
                self.funn = 'PlantePlass'
                if self.debug: print(self.funn)

    def find_sea_star(self):
        # 1 Maskerer ut røde farger.
        limits = HSV_colors.get('rød')
        mask = cv2.inRange(self.hsv, limits[0], limits[1])
        if self.debug: show(mask)

        # 2 Finner stjerne ved å sette krav til "kompakthet" og areal.:
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours.sort(key=cv2.contourArea, reverse=True)  # Sortert i synkende rekkefølge

        if len(contours) != 0 and cv2.contourArea(contours[0]) != 0:
            areal_convex = cv2.contourArea(cv2.convexHull(contours[0]))
            areal = cv2.contourArea(contours[0])
            solidity = areal / areal_convex
            if self.sea_star_min_solidity < solidity < self.sea_star_max_solidity \
                    and self.sea_star_min_areal < areal < self.sea_star_max_areal:
                self.funn = 'Stjerne'
                if self.debug: print(self.funn)

    def find_sponge(self):
        # Sjekk ut Sort/ find circle for å finne ring
        limits = HSV_colors.get('svamp')
        mask = cv2.inRange(self.hsv, limits[0], limits[1])
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        if self.debug: show(mask)
        # if self.debug: show(gray)
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours.sort(key=cv2.contourArea, reverse=True)  # Sortert i synkende rekkefølge
        if len(contours) != 0 and cv2.contourArea(contours[0]) != 0:
            areal_convex = cv2.contourArea(cv2.convexHull(contours[0]))
            areal = cv2.contourArea(contours[0])
            solidity = areal / areal_convex
            (x, y), radius = cv2.minEnclosingCircle(contours[0])
            areal_circle = 3.14 * radius ** 2
            fill_of_circle = areal / areal_circle
            _, _, wdt, hgt = cv2.boundingRect(contours[0])
            if self.sponge_min_areal < areal < self.sponge_max_areal and (fill_of_circle > self.spong_min_fill_of_cicle
                                                                          or (
                                                                                  self.sponge_min_soliditet < solidity and self.spong_min_width < wdt < self.spong_max_width
                                                                                  and self.spong_min_height < hgt < self.spong_max_height)):
                self.funn = 'Svamp'
                '''
                ##############################
                # dp -  This parameter is the inverse ratio of the accumulator resolution to the image resolution (see Yuen et al. for more details).
                #       Essentially, the larger the dp gets, the smaller the accumulator array gets.
                # minDist - minimum avstand mellom senter i en ring til senter i en annen ring
                # param1 - Gradient values used to handle edge detection Yuen et al. method.
                # param2 - Accumulator threshold value for the cv2.HOUGH_GRADIENT method.
                #          The smaller the threshold is, the more circles will be detected (including false circles).
                #          The larger the threshold is, the more correct circles will potentially be returned
                #####################################
                circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 2, 20,
                                   param1=30, param2=20, minRadius=8, maxRadius=20)
                

                if circles is not None:
                    self.funn = 'Svamp'
                '''
                if self.debug:
                    print(self.funn)
                    # Tegner sirkelen inn i originalbildet
                    cv2.circle(self.img, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    '''
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        # draw the outer circle
                        cv2.circle(self.img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        # draw the center of the circle
                        cv2.circle(self.img, (i[0], i[1]), 2, (0, 0, 255), 3)
                    '''
                    show(self.img)
                # pshow(self.hsv)

    def find_coral_reef(self):
        # Alternativ 1: finner "horisontalt" sort ben => sikkert koralrev
        limits = HSV_colors.get('sort')
        mask = cv2.inRange(self.hsv, limits[0], limits[1])
        if self.debug: show(mask)

        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours.sort(key=cv2.contourArea, reverse=True)  # Sortert i synkende rekkefølge
        if len(contours) != 0 and cv2.contourArea(contours[0]) != 0:
            rect = cv2.minAreaRect(contours[0])
            areal_min_area_rect = rect[1][1] * rect[1][0]  # Høyde * Bredde
            bnd_bx = cv2.boundingRect(contours[0])
            wdt = bnd_bx[2]
            hgt = bnd_bx[3]
            areal = cv2.contourArea(contours[0])
            extent = areal / areal_min_area_rect

            # Alternativ 1 - Full lengde horisontalt rør med T formet endestykke.
            if self.coral_reef_alt_1_min_height < hgt < self.coral_reef_alt_1_maks_height and self.coral_reef_alt_1_min_width < wdt < self.coral_reef_alt_1_maks_width and self.coral_reef_alt_1_min_extent < extent < self.coral_reef_alt_1_max_extent:
                self.funn = 'Koralrev'
                if self.debug: print(self.funn)
            # Alternativ 2 - Full lengde enkelt vertikalt rør.
            elif self.coral_reef_alt_2_min_height < hgt <= self.coral_reef_alt_2_maks_height and self.coral_reef_alt_2_min_width < wdt < self.coral_reef_alt_2_maks_width and self.coral_reef_alt_2_min_extent < extent:
                self.funn = 'Koralrev'
                if self.debug: print(self.funn)
            # Alternativ 3 - Hjørnestykke eller avkuttet horisontalt rør med T formet endestykke.
            elif self.coral_reef_alt_3_min_height < hgt < self.coral_reef_alt_3_maks_height and self.coral_reef_alt_3_min_width < wdt < self.coral_reef_alt_3_maks_width and self.coral_reef_alt_3_min_extent < extent < self.coral_reef_alt_3_max_extent:
                self.funn = 'Koralrev'
                if self.debug: print(self.funn)
            # Alternativ 4 - Stor T-formet fot som ikke har blitt overskygget
            elif self.coral_reef_alt_4_maks_areal > areal > self.coral_reef_alt_4_min_areal and self.coral_reef_alt_4_min_width < wdt and self.coral_reef_alt_4_min_height < hgt and self.coral_reef_alt_4_max_extent > extent > self.coral_reef_alt_4_min_extent:
                self.funn = 'Koralrev'
                if self.debug: print(self.funn)


class Seabed:
    def __init__(self, img, map, debug=True, debug_square=False, inspection=True):
        self.inspection = inspection
        self.debug_square = debug_square
        self.debug = debug
        self.map = map
        self.img_orig = img
        self.img = None
        self.img_rotert = None
        self.img_hsv = None
        self.hsv_median_blur_kernel = 9
        self.img_roi = None
        self.img_roi_hsv = None
        self.std_img_height = 1000
        self.blue_pipe_min_height = 750
        self.std_roi_size = (270, 810)  # (rader, kolonner)
        self.std_roi_center = 0  # (500,500)  # Uviktig omtrentlig tall
        self.squares = []
        self.blue_pipe_width_red = None
        self.blue_pipe_extension = 0  # 14 piksler justering, manuell
        self.blue_pipe_diff_lim = 150
        self.min_areal_blue_line_contours = 100
        self.roi_corner_points = None
        self.initial_image_transformations()

    def main(self):
        self.find_roi()
        self.split_roi_into_squares()
        return self.draw_map()

    def initial_image_transformations(self):
        try:
            # Rotate image if aspect ratio is more than 1.
            aspect_ratio = self.img_orig.shape[1] / self.img_orig.shape[0]
            if aspect_ratio > 1:
                self.img_orig = cv2.rotate(self.img_orig, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Scaling factor.
            f = 1 - (self.img_orig.shape[0] - self.std_img_height) / self.img_orig.shape[0]
            self.img = cv2.resize(self.img_orig, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

            # Transforming from BGR to HSV format.
            self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            # Blurring of hsv image.
            self.img_hsv = cv2.medianBlur(self.img_hsv, self.hsv_median_blur_kernel)
        except Exception as e:
            print("Oops!", e.__class__, "occurred in Initaial transformations .")
            print(e.args)

    def find_roi(self):

        ## 1 Henter inn definert farge
        limits = HSV_colors.get('blå')

        ## 2 Henter ut pixler som har definert farge og lager maskering
        mask = cv2.inRange(self.img_hsv, limits[0], limits[1])
        #######################################################################################################
        ## 3 Leter etter blå linjer i bildet.
        #       - Hvis en eller to av de to største konturene ikke er lengre en definert lenge, brukes en lukkende metode.
        #       - Dette gjentas til de to største konturene er minimum definert lengde lange.
        #       -

        counter = 1
        kernel = np.ones((20, 1), np.uint8)  # Vertikal retning.
        rotated = True  # Rotere bildet første gang
        while True:
            # 3.1 Henter ut konturene og sortere dem etter areal
            contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours.sort(key=cv2.contourArea, reverse=True)  # Sortert i synkende rekkefølge etter areal.
            '''
            # 3.1.1 Filtrere konturere etter enkel areal grense:
            large = []
            for c in contours:
                if cv2.contourArea(c) > self.min_areal_blue_line_contours:
                    large.append(c)
            contours = large
            '''
            # 3.2 Henter ut høyden til hver kontur i antall piksler
            height1 = cv2.boundingRect(contours[0])[3]
            height2 = cv2.boundingRect(contours[1])[3]
            # 3.3 Justerer minste lengde til rørene.
            # Denne biten gjør det mindre sannsynlig at vi ender opp med rør av froskjellige lengder,
            if max(height1, height2) > (self.blue_pipe_diff_lim + min(height1, height2)):
                self.blue_pipe_min_height = max(height1, height2) - self.blue_pipe_diff_lim

            # 3.4 Rotere bildet:
            #       - Lettere med morforlogi når konturene er vertikale
            #       - Find corner points trenger mest mulig rett firkant ift koordinatsystemet.
            if rotated:
                angles = []
                for i in range(0, 2):
                    [vx, vy, x, y] = cv2.fitLine(contours[i], cv2.DIST_HUBER, 0, 0.01, 0.01)
                    angles.append(calculate_pipe_angle(contours[i]))
                ang = sum(angles) / 2
                if ang < 90:
                    angel_to_rotate = (90 - ang) * (-1)
                elif ang < 225:
                    angel_to_rotate = 180 - ang
                else:
                    angel_to_rotate = ang - 270

                # Roterer bildene vi arbeider med
                mask = rotation(mask, angel_to_rotate)
                self.img = rotation(self.img, angel_to_rotate)
                self.img_hsv = rotation(self.img_hsv, angel_to_rotate)

                # Oppdatere konturene etter rotasjonen:
                contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
                contours.sort(key=cv2.contourArea, reverse=True)

                rotated = False

                if self.debug:
                    print('Rotasjons Vinkel: {:+.1f}'.format(angel_to_rotate))
                    print('Snitt vinkel: {:+.1f}'.format(ang))


            # DEBUG Tegner inn og viser de to største konturene
            if self.debug:
                empty = np.zeros(self.img_hsv.shape[:2], dtype=np.uint8)  # Alternativt: emtpy = np.zeros_like(mask)
                empty = cv2.drawContours(empty, [contours[0]], 0, color=255, thickness=-1)
                #show(empty)
                empty = cv2.drawContours(empty, [contours[1]], 0, color=255, thickness=-1)
                show(empty)
            # 3.3 Hvis begge er høye nok er vi fornøyde, hvis ikke må vi arbeide mer med bilde
            if height1 > self.blue_pipe_min_height and height2 > self.blue_pipe_min_height:
                forskyvning = (counter - 1) * 2
                for p in contours[0]:
                    p[0][1] -= forskyvning
                for p in contours[1]:
                    p[0][1] -= forskyvning
                #if self.debug:
                    #show(mask)
                    # moment = cv2.moments(contours[0])
                    # center = [int(moment['m10'] / (moment['m00'] + 1e-6)), int(moment['m01'] / (moment['m00'] + 1e-6))]
                    # print(center)
                break
            else:
                # 1.3.3 Ved og bruke utvide først, for så å redusere, får vi en "lukkende" effekt på konturene.
                #       - På denne måten får vi koblet sammen de blå linjene som av en ukjent grunn er brutt.
                mask = cv2.dilate(mask, kernel, iterations=counter)
                mask = cv2.erode(mask, kernel, iterations=counter)

                counter += 1
                # DEBUG viser resultatet etter behandling
            #if self.debug:
                # show(mask)
                # contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
                # contours.sort(key=cv2.contourArea, reverse=True)
                # moment = cv2.moments(contours[0])
                # center = [int(moment['m10'] / (moment['m00'] + 1e-6)), int(moment['m01'] / (moment['m00'] + 1e-6))]
                # print(center)


        ##################################################################################
        ## 4 Går gjennom alle konturene og plukker ut de som er større en definert grense:
        point_list = []
        width_list = []
        for i in range(2):
            # 4.1 Finner minste rektangel som kan omkranse gjeldende kontur.
            #       - minAreaRect Format: (center(x,y), (width, height), angle of rotation).
            rekt = cv2.minAreaRect(contours[i])

            # 4.2 Gjør om formatet til "rect" til et "Contour" format
            boks = cv2.boxPoints(rekt)
            boks = np.int0(boks)

            # DEBUG Tegner inn boksen til konturen, samt fylt kontur.
            if self.debug:
                # cv2.drawContours(self.img, [boks], 0, (0, 255, 0), 2)
                cv2.drawContours(self.img, [contours[i]], 0, color=255, thickness=-1)
                #show(self.img)

            # 4.3 Legger hjørnepunktene fra gjeldene boks til i en punktliste
            #       Legger også til bredden på minAreaRect boksen.
            point_list.append(boks.tolist())
            width_list.append(min(rekt[1][0], rekt[1][1]))

        # 4.4 Formaterer punktlisten til et til et "Contour" format
        point_list = np.array(point_list, dtype=np.int64)
        point_list.resize(8, 2)
        self.blue_pipe_width_red = sum(width_list) / 4
        # 4.5 Passer på at ingen av kordinatene er negative, gjør isåfall disse til 0:
        for p in point_list:
            if p[0] < 0:
                p[0] = 0
            if p[1] < 0:
                p[1] = 0

        ## 5 Finner rektanglet/roi(Region of Interest) definert av de to blå rørene.
        #    - Klipper rundt funnet rektangel og roterer bildet slik at det står vertikalt.

        # DEBUG Finner minste rektangel definert av alle hjørne punktene
        #      - Formatere til "Contour" format
        #      - Tegner inn boksen til ROI omerådet vi har funnet.

        self.roi_corner_points = find_corner_points(point_list, self.std_roi_center)
        if self.debug:
            rekt = cv2.minAreaRect(self.roi_corner_points)  # ( center (x,y), (width, height), angle of rotation )
            boks = cv2.boxPoints(rekt)  # Format med fire hjørnepunkter
            boks = np.int0(boks)  #
            # cv2.drawContours(self.img, [boks], 0, (0, 255, 0), 2)
            for x in self.roi_corner_points:
                cv2.circle(self.img, (int(x[0]), int(x[1])), 5, (255, 255, 0), 2)
            #show(self.img)

        self.roi_corner_points = shrink_contour(self.roi_corner_points, y_axis=self.blue_pipe_extension,
                                                x_axis=self.blue_pipe_width_red)
        if self.debug:
            rekt = cv2.minAreaRect(self.roi_corner_points)  # ( center (x,y), (width, height), angle of rotation )
            boks = cv2.boxPoints(rekt)  # Format med fire hjørnepunkter
            boks = np.int0(boks)  #
            # cv2.drawContours(self.img, [boks], 0, (0, 255, 0), 2)
            for x in self.roi_corner_points:
                cv2.circle(self.img, (int(x[0]), int(x[1])), 5, (0, 255, 255), 2)
            show(self.img)

        self.img_roi_hsv = change_perspective_and_crop(self.img_hsv, self.roi_corner_points, self.std_roi_size[0],
                                                       self.std_roi_size[1])

        self.img_roi = change_perspective_and_crop(self.img, self.roi_corner_points, self.std_roi_size[0],
                                                   self.std_roi_size[1])

        #if self.debug:
        #    show(np.hstack((self.img_roi, self.img_roi_hsv)))
        #if self.inspection:
        #    show(np.hstack((self.img_roi, self.img_roi_hsv)))

    def split_roi_into_squares(self):
        try:
            total_height = int(self.img_roi.shape[0])
            height = int(total_height / 9)  # Høyde for en rute
            total_width = int(self.img_roi.shape[1])
            width = int(total_width / 3)  # Bredde for en rute
            x = 0
            y = 0

            if self.debug or self.inspection: e = np.zeros_like(self.img_roi)
            # Går gjennom alle kordinator og oppretter objekter av klassen Ruter for hele rutenettet.
            for i in range(1, 10):
                for j in range(1, 4):

                    if j == 1 and i == 5:
                        a = 0

                    # Klipper ut et lite bilde av hver enkelt rute
                    roi = self.img_roi[y:y + height, x:x + width]
                    hsv = self.img_roi_hsv[y:y + height, x:x + width]
                    # Oppretter objekter av klassen Square
                    sq = Square(roi, hsv, [j, i], [[x, x + width], [y, y + height]], debug=self.debug_square)
                    sq.scan_for_objects()
                    self.squares.append(sq)

                    # DEBUG Viser frem en og en rute.

                    if self.debug_square:
                        e[y:y + height - 1, x:x + width - 1] = self.img_roi[y:y + height - 1, x:x + width - 1]
                        show(e)
                        # pshow(rute)
                    if self.inspection:
                        e[y:y + height - 1, x:x + width - 1] = self.img_roi[y:y + height - 1, x:x + width - 1]
                    x += width
                x = 0
                y += height

            if self.inspection:
                show(e)
        except Exception as e:
            print("Oops!", e.__class__, "occurred in Split ROI into squares .")

    def draw_map(self):
        try:
            # Diverse start imformasjon
            reduksjon = 20  # piksler
            brd = [[188, 528], [63, 1190]]  # [(x1,x2),(y1,y2)] # Innterresang del av kartbildet
            hoyde = brd[1][1] - brd[1][0]
            bredde = brd[0][1] - brd[0][0]
            dx = bredde / 3
            dy = hoyde / 9
            radius = int(((dx + dy) / 4) - reduksjon / 2)
            koral = []
            # Går gjennom alle rutene og tegner inn symbol for hvert enkelt funn.
            for sq in self.squares:
                if sq.funn == 'Svamp':
                    x = int((sq.coordinate[0] - 0.5) * dx + brd[0][0])
                    y = int((sq.coordinate[1] - 0.5) * dy + brd[1][0])
                    cv2.circle(self.map, (x, y), radius, (0, 255, 0), 3)
                elif sq.funn == 'Stjerne':
                    x = int((sq.coordinate[0] - 0.5) * dx + brd[0][0])
                    y = int((sq.coordinate[1] - 0.5) * dy + brd[1][0])
                    cv2.circle(self.map, (x, y), radius, (255, 0, 0), 3)
                elif sq.funn == 'PlantePlass':
                    x = int((sq.coordinate[0] - 0.5) * dx + brd[0][0])
                    y = int((sq.coordinate[1] - 0.5) * dy + brd[1][0])
                    cv2.circle(self.map, (x, y), radius, (0, 255, 255), 3)
                elif sq.funn == 'Koralrev':
                    koral.append(sq)

            # Spesial tilfelle for koralrev. Tegner her inn en ellipse.
            # Kode finner størrelse, orientering og plassering som ellipsen skal ha.

            x, y = [0, 0]

            colonne_1 = []
            colonne_2 = []
            colonne_3 = []

            for sq in koral:
                if sq.coordinate[0] == 1:
                    colonne_1.append(sq)
                elif sq.coordinate[0] == 2:
                    colonne_2.append(sq)
                else:
                    colonne_3.append(sq)

            if len(colonne_1) > len(colonne_2) and len(colonne_1) > len(colonne_3):
                colonne_1.sort(key=lambda square: square.coordinate[1])  # Stigende rekkefølge y koordinater
                y_cord = 0
                failed = []
                for i in range(len(colonne_1)):
                    if y_cord == 0:
                        y_cord = colonne_1[i].coordinate[1]
                    elif colonne_1[i].coordinate[1] != (y_cord + 1) and colonne_1[i].coordinate[1] != (y_cord + 2):
                        failed.append(colonne_1[i - 1])
                        y_cord = colonne_1[i].coordinate[1]
                    else:
                        y_cord = colonne_1[i].coordinate[1]
                for sq in failed:
                    colonne_1.remove(sq)
                for sq in colonne_1:
                    x += sq.coordinate[0]
                    y += sq.coordinate[1]
                H = len(colonne_1)
                B = 1
            elif len(colonne_2) > len(colonne_1) and len(colonne_2) > len(colonne_3):
                colonne_2.sort(key=lambda square: square.coordinate[1])  # Stigende rekkefølge y koordinater
                y_cord = 0
                failed = []
                for i in range(len(colonne_2)):
                    if y_cord == 0:
                        y_cord = colonne_2[i].coordinate[1]
                    elif colonne_2[i].coordinate[1] != (y_cord + 1) and colonne_2[i].coordinate[1] != (y_cord + 2):
                        failed.append(colonne_2[i - 1])
                        y_cord = colonne_2[i].coordinate[1]
                    else:
                        y_cord = colonne_2[i].coordinate[1]
                for sq in failed:
                    colonne_2.remove(sq)
                for sq in colonne_2:
                    x += sq.coordinate[0]
                    y += sq.coordinate[1]
                H = len(colonne_2)
                B = 2
            else:
                colonne_3.sort(key=lambda square: square.coordinate[1])  # Stigende rekkefølge y koordinater
                y_cord = 0
                failed = []
                for i in range(len(colonne_3)):
                    if y_cord == 0:
                        y_cord = colonne_3[i].coordinate[1]
                    elif colonne_3[i].coordinate[1] != (y_cord + 1) and colonne_3[i].coordinate[1] != (y_cord + 2):
                        failed.append(colonne_3[i - 1])
                        y_cord = colonne_3[i].coordinate[1]
                    else:
                        y_cord = colonne_3[i].coordinate[1]
                for sq in failed:
                    colonne_3.remove(sq)
                for sq in colonne_3:
                    x += sq.coordinate[0]
                    y += sq.coordinate[1]
                H = len(colonne_3)
                B = 3

            b = int((1 * dx - reduksjon) / 2)  # Piksel bredde til figur
            h = int((H * dy - reduksjon) / 2)  # Pikse høyde til figur
            x = int(((B) - 0.5) * dx + brd[0][0])  # senter av figur
            y = int(((y / H) - 0.5) * dy + brd[1][0])  # senter av figur

            cv2.ellipse(self.map, (x, y), (b, h), 0, 0, 360, (0, 0, 255), 3)

            self.map = cv2.cvtColor(self.map, cv2.COLOR_BGR2RGB)
            self.map = cv2.resize(self.map, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)

            return self.map
        except Exception as e:
            print("Oops!", e.__class__, "occurred in Draw map .")
            print(e.args)


def find_corner_points(points, center=0, thresh=5):
    '''Returns the four extramal cornerpoints out of list of points defining a square.
        :points - List of points
        :center - Defined senter of rectangle
        :thres - differential required to choose next corner over the existing choise
    '''
    if center == 0:
        rect = cv2.minAreaRect(points)
        center = rect[0]
    th = thresh
    left_high = center
    left_low = center
    right_high = center
    right_low = center

    for p in points:
        if p[0] < center[0] and p[1] < center[1] and ((left_high[0] - p[0]) +
                                          (left_high[1] - p[1]) > th):
            left_high = [p[0], p[1]]
        if p[0] < center[0] and p[1] > center[1] and ((left_low[0] - p[0]) +
                                          (p[1] - left_low[1]) > th):
            left_low = [p[0], p[1]]
        if p[0] > center[0] and p[1] < center[1] and ((right_high[1] - p[1]) +
                                          (p[0] - right_high[0]) > th):
            right_high = [p[0], p[1]]
        if p[0] > center[0] and p[1] > center[1] and ((p[0] - right_low[0]) +
                                          (p[1] - right_low[1]) > th):
            right_low = [p[0], p[1]]

    new_points = np.array([left_high, left_low,
                           right_low, right_high], dtype = np.float32)
    return new_points


def change_perspective_and_crop(img, source_points, new_width, new_height):
    ''':source_points
                - Four source points that will be moved.
                - Start in top left corner, go counter clockwise arond the square.
        :new_width
                - Width of produced image
        :new_heigt
                - Height of produced image
        :img
                - Input image
    '''
    height = new_height
    width = new_width
    src_pts = source_points
    dst_pts = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)
    perspective_m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img, perspective_m, (width, height))
    return result


def shrink_contour(points, y_axis=0, x_axis=0):
    # Shrink towards center of contour. No weighting. # p[1]=y p[0]=x
    moment = cv2.moments(points)
    center = [int(moment['m10'] / (moment['m00'] + 1e-6)), int(moment['m01'] / (moment['m00'] + 1e-6))]
    for p in points:
        if p[1] < center[1] and p[0] < center[0]:  # Upper Left
            p[1] += y_axis
            p[0] += x_axis
        elif p[1] > center[1] and p[0] > center[0]:  # Lower Right
            p[1] -= y_axis
            p[0] -= x_axis
        elif p[1] < center[1] and p[0] > center[0]:  # Upper Right
            p[1] += y_axis
            p[0] -= x_axis
        else:  # Lower left
            p[1] -= y_axis
            p[0] += x_axis
        np.float32(p[1])
        np.float32(p[0])
    return points


def adjust_contour_position(points, y_axis=0, x_axis=0):
    for p in points:
        p[0][0] += x_axis
        p[0][1] += y_axis
    return points


def calculate_pipe_angle(contour):
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_HUBER, 0, 0.01, 0.01)
    # Format til linjen
    # Vx og Vy x og y komponenter til en normaliserte vektor. Altså retningsvektoren
    # x og y er komponentene til forskyvningsvektoren.
    ang = np.rad2deg(np.arctan2(vy[0], vx[0]))
    if ang < 0:
        ang += 360
    return ang


def rotation(img, angle):
    rows, cols = img.shape[:2]
    matrise = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)
    rst = cv2.warpAffine(img, matrise, (cols, rows))
    return rst


def translation(img, x, y):
    rows, cols = img.shape[:2]
    matrise = np.float32([[1, 0, x], [0, 1, y]])
    rst = cv2.warpAffine(img, matrise, (cols, rows))
    return rst


def pshow(img):
    plt.imshow(img)
    plt.show()


def show(img, name='standard'):
    cv2.imshow(name, img)
    cv2.waitKey()


def test_ruter():
    test2 = []
    for i in range(1, 101):
        img = cv2.imread(f'Github/Eksperimentelt/Bilder/havbunn/rute_test/test_ ({i}).png')
        img = cv2.resize(img, (90, 90), interpolation=cv2.INTER_AREA)
        show(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.medianBlur(hsv, 5)
        sq = Square(img, hsv, debug=True)
        sq.scan_for_objects()

        print(f'Nr:{i} - {sq.funn}')

        if sq.funn == None:
            test2.append(sq)
    for sq in test2:
        show(sq.img)
        sq.scan_for_objects()


def test_circle():
    map = cv2.imread('Github/Eksperimentelt/Bilder/havbunn/havbunn.png')
    map = cv2.rotate(map, cv2.ROTATE_90_CLOCKWISE)

    img = cv2.imread('Github/Eksperimentelt/Bilder/havbunn/IMG_svamp_2.jpg')
    a = Seabed(img, map, debug=False)
    a.find_roi()
    img = a.img_roi
    hsv = a.img_roi_hsv
    sq = Square(img, hsv, debug=True)
    sq.find_sponge()


def test_shr_cont():
    o = cv2.imread('Github/Eksperimentelt/Bilder/Konteiner/mate_1.png')
    a = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
    _, b = cv2.threshold(a, 170, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    contours = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours.sort(key=cv2.contourArea, reverse=True)  # Sortert i synkende rekkefølge
    rekt = cv2.minAreaRect(contours[0])
    boks = cv2.boxPoints(rekt)
    boks = np.int0(boks)

    cv2.drawContours(o, [boks], 0, (0, 255, 0), 2)
    print(a.shape)
    show(o)

    shrink_contour(boks, x_axis=30)
    cv2.drawContours(o, [boks], 0, (0, 0, 255), 2)
    print(a.shape)
    show(o)

    shrink_contour(boks, y_axis=30)
    cv2.drawContours(o, [boks], 0, (255, 0, 0), 2)
    show(o)
    print(a.shape)
    shrink_contour(boks, 50, 50)
    cv2.drawContours(o, [boks], 0, (255, 0, 0), 2)
    show(o)
    print(a.shape)
    shrink_contour(boks, -20, -20)
    cv2.drawContours(o, [boks], 0, (255, 0, 0), 2)
    show(o)
    print(a.shape)


def test_persp_chg():
    map = cv2.imread('Github/Eksperimentelt/Bilder/havbunn/havbunn.png')
    map = cv2.rotate(map, cv2.ROTATE_90_CLOCKWISE)

    img = cv2.imread('Github/Eksperimentelt/Bilder/havbunn/IMG_svamp_3.jpg')
    a = Seabed(img, map, debug=False)
    a.main()
    p = a.roi_corner_points
    for x in p:
        cv2.circle(a.img, (x[0], x[1]), 5, (255, 255, 0), 2)
    pshow(a.img)

    height = 810
    width = 270
    src_pts = p
    dst_pts = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)
    perspective_m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(a.img, perspective_m, img.shape[:2])[0:height, 0:width]
    img = cv2.warpPerspective(a.img, perspective_m, img.shape[:2])

    show(result)
    print(result.shape)
    show(img)
    print(img.shape)
    img = change_perspective_and_crop(a.img, p, 270, 810)
    show(img)
    print(img.shape)


def test_programm():
    img = cv2.imread(f'Bildebehandling/Bilder/Oversikt_havbunn/Oversiktsbilder/test_ ({5}).jpg')
    map = cv2.imread('Bildebehandling/Bilder/Oversikt_havbunn/havbunn.png')
    map = cv2.rotate(map, cv2.ROTATE_90_CLOCKWISE)
    a = Seabed(img, map, debug=False, debug_square=False, inspection=False)
    a.initial_image_transformations()
    a.find_roi()
    a.split_roi_into_squares()
    return cv2.cvtColor(a.draw_map(), cv2.COLOR_BGR2RGB)


def test_oversikt():
    snitt = []
    for i in range(1, 41):

        img = cv2.imread(f'Bildebehandling/Bilder/Oversikt_havbunn/Oversiktsbilder/test_ ({i}).jpg')
        map = cv2.imread('Bildebehandling/Bilder/Oversikt_havbunn/havbunn.png')
        t1 = time.perf_counter()
        map = cv2.rotate(map, cv2.ROTATE_90_CLOCKWISE)
        a = Seabed(img, map, debug=False , debug_square=False, inspection = False)
        b = None
        b = a.main()
        tid = time.perf_counter()-t1
        snitt.append(tid)
        #if b is not None:
            #show(b)
        print(f'Nr: {i} , Tid {round(tid*1000)} [ms]')
    print(f"Snitttid: {round(sum(snitt)/len(snitt)*1000)}")


if __name__ == '__main__':
    # test_circle()
    # test_ruter()
    test_oversikt()
    # test_shr_cont()
    # test_persp_chg()
    # test_programm()

    '''
    map = cv2.imread('Github/Eksperimentelt/Bilder/havbunn/havbunn.png')
    map = cv2.rotate(map, cv2.ROTATE_90_CLOCKWISE)

    img = cv2.imread('Github/Eksperimentelt/Bilder/havbunn/IMG_svamp_3.jpg')
    a = Seabed(img, map, debug=True)
    a.main()
    '''
