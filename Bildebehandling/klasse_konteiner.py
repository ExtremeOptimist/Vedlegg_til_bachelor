import time
import traceback
from sklearn.cluster import KMeans
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
from matplotlib import pyplot as plt
from Bildebehandling.klasse_havbunn import shrink_contour, find_corner_points


# OBS OBS HSV formatet i openCV har H-hue verdier fra 0-179. I motsetning 0-360 som er standard.
HSV_colors = {
    'rød': [np.array([0, 20, 0], dtype='uint8'), np.array([60, 255, 255], dtype='uint8')],
    'gul': [np.array([61, 20, 0], dtype='uint8'), np.array([120, 255, 255], dtype='uint8')],
    'grønn': [np.array([121, 20, 0], dtype='uint8'), np.array([180, 255, 255], dtype='uint8')],
    'turkus': [np.array([181, 20, 0], dtype='uint8'), np.array([240, 255, 255], dtype='uint8')],
    'blå': [np.array([241, 20, 0], dtype='uint8'), np.array([300, 255, 255], dtype='uint8')],
    'lilla': [np.array([301, 20, 0], dtype='uint8'), np.array([360, 255, 255], dtype='uint8')],
    'issoler_farger': [np.array([0, 70, 0], dtype='uint8'), np.array([179, 255, 255], dtype='uint8')],
    'issoler_farger_2': [np.array([0, 30, 0], dtype='uint8'), np.array([179, 255, 255], dtype='uint8')],
    'finn_teip_mate': [np.array([0, 95, 60], dtype='uint8'), np.array([179, 255, 255], dtype='uint8')],
    'finn_teip': [np.array([0, 150, 60], dtype='uint8'), np.array([179, 255, 255], dtype='uint8')]
}


class Side:
    def __init__(self, img, debug=True):
        self.debug = debug  # Se utviklingen til bildet gjennom prosessen.
        self.min_areal = 1000  # Minimums tall på piksler for å kodkjenne kantmarkør
        self.piksler_i_hoyden = 200  # Antall piksler i høyden på ferdig bildet.
        self.forhold = None  # Objektes forhold mellom lengde og bredde.
        self.type = None  # Topp, langside eller kortside
        self.img = img  # Orginalbildet, størrelsen er redusert
        self.img_bgr = None  # Behandlet bilde i BGR format.
        self.img_hsv = None  # Behandlet bilde i HSV format.
        self.constraints = {}  # Dictionary med oversikt over relasjon mellom farge og kant.
        self.regions_of_side = None  # Definerte pikselregioner for tape merking av sideflate.
        self.median_blur_kernel = 9

        self.initial_image_transformations()

        #self.test()
        #self.define_attributes()        # Bestemme fargekoder på kantene

    def initial_image_transformations(self):
        try:
            # 1: Roter bildet hvis størrelsesforholdet er mindre enn en. Vi skal ha "liggende" bilder.
            storrelsesforgold = self.img.shape[1] / self.img.shape[0]
            if storrelsesforgold < 0.8: # 0.8 pga bruk av testbilder, Unødvedig ved bruk av ROV
                self.img = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # 2: Skallerer bildet til standard størrelse, 2x ferdig størrelse.
            f = 1 - (self.img.shape[0] - self.piksler_i_hoyden*2) / self.img.shape[0]
            self.img = cv2.resize(self.img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

            # 3: Filtrerer/jevner ut bildet med medianBlur.
            self.img_bgr = cv2.medianBlur(self.img, self.median_blur_kernel)

            # 4: Oppretter et bilde i HSV format, dette er flitert slik som img_bgr.
            self.img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)


        except Exception as e:
            print("Oops!", e.__class__, "occurred in Initaial transformations .")
            print(e.args)
            traceback.print_exc()

    def find_box_and_crop_img(self):
        try:
            # 1. Issoler alle fargene i bildet.
            limits = HSV_colors.get('finn_teip')
            mask = cv2.inRange(self.img_hsv, limits[0], limits[1])

            # 2. Finn konturene og koble sammen alle de som er store nok.
            contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours.sort(key=cv2.contourArea, reverse=True)
            empty = np.zeros_like(mask)

            large = []
            x1, y1 = [0, 0]
            for c in contours:
                if cv2.contourArea(c) > self.min_areal:
                    # Tegner de store konturene inn i det tomme bildet.
                    empty = cv2.drawContours(empty, [c], 0, color=255, thickness=-1)
                    # Binder sammen konturene ved å tegne en strek mellom dem.
                    (x2, y2), _, _ = cv2.minAreaRect(c)
                    x2 = np.int0(x2)
                    y2 = np.int0(y2)
                    if len(large) > 0:
                        cv2.line(empty, (x1, y1), (x2, y2), 255, 15)
                    x1, y1 = [x2, y2]
                    large.append(c)
                    if self.debug:
                        show(empty)


            # 4. Finner den store sammenkoblede konturen, og dens minste omkransende rektangel.
            contours = cv2.findContours(empty, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours.sort(key=cv2.contourArea, reverse=True)

            # 5. Bestemmer sideflatens hjørnepunkter.
            rekt = cv2.minAreaRect(contours[0])
            boks = cv2.boxPoints(rekt)
            boks = shrink_contour(boks, y_axis=-12, x_axis=-18) # Utvider boksen.
            boks = find_corner_points(boks) # Sortere punktene i riktig rekkefølge.
            if self.debug:
                boks = np.int0(boks)

                self.img_bgr = cv2.drawContours(self.img_bgr, [boks], 0, (0,0,250), 2)
                for p in boks:
                    self.img_bgr = cv2.circle(self.img_bgr,(int(p[0]),int(p[1])),10,(250,20,2),2)
                show(self.img_bgr)
            boks = np.float32(boks)

            # Utifra rektangelsets rotasjon bestemmer vi hva som blir den nye bilde høyden og bredden.
            if abs(rekt[2]) < 40:
                f = 1 - (rekt[1][1] - self.piksler_i_hoyden) / rekt[1][1]
                hgt = np.float32(rekt[1][1] * f)
                wdt = np.float32(rekt[1][0] * f)
            else:
                f = 1 - (rekt[1][0] - self.piksler_i_hoyden) / rekt[1][0]
                wdt = np.float32(rekt[1][1] * f)
                hgt = np.float32(rekt[1][0] * f)

            self.forhold = wdt / hgt    # Forholdet defineres utifra boksens høyde og bredde.

            # 6. Klipper ut sideflaten, og justerer den til standarisert format.
            self.img_hsv = change_perspective_and_crop(self.img_hsv, boks, wdt, hgt)

            # 7. Gjør om fra HSV til BGR bildet.
            self.img_bgr = cv2.cvtColor(self.img_hsv, cv2.COLOR_HSV2BGR)

            # 8: Definere fargeteipenes plassering for de forskjellige regionene.
            #      Disse er anderledes for kortside/langside.
            #      Format: [[min-x, maks-x],[min-y, maks-y]]
            if self.forhold > 1.5:
                self.regions_of_side = {
                    'left': [[0, 50], [0, 200]],
                    'right': [[375, 500], [0, 200]],
                    'top': [[50, 400], [0, 50]],
                    'bottom': [[50, 400], [150, 200]]
                }
            else:
                self.regions_of_side = {
                    'left': [[0, 50], [0, 200]],
                    'right': [[150, 200], [0, 200]],
                    'top': [[50, 150], [0, 50]],
                    'bottom': [[50, 150], [150, 200]]
                }

            if self.debug:
                show(self.img_bgr)
        except Exception as e:
            print("Oops!", e.__class__, "occurred in find box and crop.")
            print(e.args)
            traceback.print_exc()

    def define_attributes(self):
                try:
                    # 1. Issoler ut alle fargene i bildet.
                    limits = HSV_colors.get('finn_teip')
                    mask = cv2.inRange(self.img_hsv, limits[0], limits[1])
                    if self.debug: show(mask)

                    # 2. Bruk erosjon og utvidelses funksjoner slik at vi står igjen med fire eller ferre konturer.
                    #           - Inkludert minstemål på konturenes størrelser.
                    teller = 1
                    kernel = np.ones((3, 3), np.uint8)
                    looping = True
                    while looping:
                        teller += 1
                        if self.debug: show(mask)
                        mask = cv2.erode(mask, kernel, iterations = teller)
                        mask = cv2.dilate(mask, kernel, iterations = teller)
                        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

                        # Fjerner alle konturer som er mindre enn minste areal
                        large = []
                        for c in contours:
                            if cv2.contourArea(c) > self.min_areal:
                                large.append(c)
                        contours = large

                        # Definerer type sideflate.
                        antall_konturer = len(contours)
                        if antall_konturer == 4:
                            looping = False
                            self.type = 'top'
                        elif antall_konturer == 3:
                            looping = False
                            if self.forhold < 1.3:
                                self.type = 'kortside'
                            else:
                                self.type = 'langside'
                        elif antall_konturer < 3:
                            raise ValueError('Fant ikke riktige konturer, juster instillingene!')

                    # Vis bildet som har maskert ut gjeldende fargeteiper.
                    if self.debug:
                        output = cv2.bitwise_and(self.img_bgr, self.img_bgr, mask = mask)
                        show(output)
                        print(self.type)

                    # 3. Finner hver enkelt konturs sin posisjon og farge. Definerer vær kant sin tilhørende farge.
                    for cnt in contours:
                        self.find_color_and_position(cnt)
                except Exception as e:
                    print("Oops!", e.__class__, "occurred in define attributes.")
                    print(e.args)
                    traceback.print_exc()

    def find_color_and_position(self, contour):
                try:
                    # 1 Finner Kordinat til gjeldene konturs sitt midtpunkt.
                    moment = cv2.moments(contour)
                    center = [int(moment['m10'] / (moment['m00'] + 1e-6)), int(moment['m01'] / (moment['m00'] + 1e-6))]

                    # 2 Lager maske for den enkelt konturen
                    empty = np.zeros(self.img_hsv.shape[:2], dtype = np.uint8)
                    cv2.drawContours(empty, [contour], 0, color = 255, thickness = -1)  # Draw filled contour in mask

                    # 3 Redusere konturene noe mer for å få mer konsistent farge.
                    kernel = np.ones((3, 3), np.uint8)
                    mask_cont = cv2.erode(empty, kernel, iterations = 1)

                    # 4 Regner ut gjennomsnittlig farge.
                    #mean_color = cv2.mean(self.img_hsv,mask=mask_cont)[0]
                    mean_color = mean_HSV_color_of_masked_area(self.img_hsv, mask_cont)

                    # 5 Definerer hvilke region konturen hører til.
                    for region in self.regions_of_side:
                        if center[0] >= self.regions_of_side[region][0][0] and center[0] <= \
                                self.regions_of_side[region][0][1]:
                            if center[1] >= self.regions_of_side[region][1][0] and center[1] <= \
                                    self.regions_of_side[region][1][1]:
                                self.constraints.update({region: mean_color})

                    if self.debug:
                        output = cv2.bitwise_and(self.img_bgr, self.img_bgr, mask = mask_cont)
                        show(output)
                except Exception as e:
                    print("Oops!", e.__class__, "occurred in find_color_and_position.")
                    print(e.args)
                    traceback.print_exc()

    def test(self):
        # 1. Issoler alle fargene i bildet.
        limits = HSV_colors.get('finn_teip')
        mask = cv2.inRange(self.img_hsv, limits[0], limits[1])

        # 2. Finn konturene og koble sammen alle de som er store nok.
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours.sort(key=cv2.contourArea, reverse=True)
        empty = np.zeros_like(mask)
        empty2 = np.zeros_like(mask)
        empty3 = np.zeros_like(mask)

        large = []
        x1,y1 = [0,0]
        pts = []
        for c in contours:
            if cv2.contourArea(c) > self.min_areal:
                empty = cv2.drawContours(empty, [c], 0, color=255, thickness=-1)
                empty2 = cv2.drawContours(empty2, [c], 0, color=255, thickness=-1)
                empty3 = cv2.drawContours(empty3, [c], 0, color=255, thickness=-1)
                (x2, y2), _, _ = cv2.minAreaRect(c)
                x2 = np.int0(x2)
                y2 = np.int0(y2)
                if len(large)>0:
                    cv2.line(empty2, (x1, y1), (x2, y2), 255, 15)
                x1,y1 = [x2,y2]
                pts.append([x2,y2])
                large.append(c)
                print(cv2.contourArea(c))
                #show(empty2)

        # Bruker Polylines til å forbinde
        pts = np.array(pts, np.int64)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(empty3, [pts], isClosed=True, color=255, thickness=5)
        #show(empty3)

        # 4. Finner den store konturen, og dens minste omkransende rektangel.
        contours = cv2.findContours(empty2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours.sort(key=cv2.contourArea, reverse=True)


        rekt = cv2.minAreaRect(contours[0])
        boks = cv2.boxPoints(rekt)
        boks = shrink_contour(boks,y_axis=-15,x_axis=-25)
        boks = find_corner_points(boks)#,[500,200])
        boks = np.float32(boks)

        if abs(rekt[2]) < 30:
            f = 1 - (rekt[1][1] - self.piksler_i_hoyden) / rekt[1][1]
            hgt = np.float32(rekt[1][1]*f)
            wdt = np.float32(rekt[1][0]*f)
        else:
            f = 1 - (rekt[1][0] - self.piksler_i_hoyden) / rekt[1][0]
            wdt = np.float32(rekt[1][1]*f)
            hgt = np.float32(rekt[1][0]*f)

        self.forhold = wdt/hgt

        self.img_hsv = change_perspective_and_crop(self.img_hsv,boks,wdt,hgt)
        self.img_bgr = cv2.cvtColor(self.img_hsv,cv2.COLOR_HSV2BGR)

        #x,y,bredde, hoyde = cv2.boundingRect(empty)

        #cv2.rectangle(self.img, (x, y), (x + bredde, y + hoyde), 255, 2)

        # 5: Forskjellige regioner for kortside/langside.
        #   Format: [[min-x, maks-x],[min-y, maks-y]]
        if self.forhold > 1.5:
            self.regions_of_side = {
                'left': [[0, 50], [0, 200]],
                'right': [[375, 500], [0, 200]],
                'top': [[50, 400], [0, 50]],
                'bottom': [[50, 400], [150, 200]]
            }
        else:
            self.regions_of_side = {
                'left': [[0, 50], [0, 200]],
                'right': [[150, 200], [0, 200]],
                'top': [[50, 150], [0, 50]],
                'bottom': [[50, 150], [150, 200]]
            }

        if self.debug:
            #cv2.drawContours(self.img, [boks], 0, (43, 111, 241), 2)
            #show(np.hstack([mask, empty2]))

            show(self.img)
            show(self.img_hsv)

            print(len(large))

    def test2(self):
        # Isollerer bort farger og finner den største konturen. Denne stilles grav til, oppfylles disse har vi en side
        limits = HSV_colors.get('issoler_farger_2')
        mask = cv2.inRange(self.img_hsv, limits[0], limits[1])
        mask = 255 - mask
        output = cv2.bitwise_and(self.img_bgr, self.img_bgr, mask=mask)
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours.sort(key=cv2.contourArea, reverse=True)


        areal_convex = cv2.contourArea(cv2.convexHull(contours[0]))
        areal = cv2.contourArea(contours[0])
        rect = cv2.minAreaRect(contours[0])
        areal_min_area_rect = rect[1][1] * rect[1][0]
        soliditet = areal/areal_convex
        ustrek = areal/areal_min_area_rect

        empty = np.zeros(self.img_hsv.shape[:2])
        empty = cv2.drawContours(empty, [contours[0]], 0, color=255, thickness=-1)
        show(empty)
        if areal > 30000 and 0.9 < soliditet and  ustrek > 0.9:
            pshow(np.hstack([self.img_hsv, output]))
        pshow(np.hstack([self.img_hsv, output]))
        '''
        gr = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, img1 = cv2.threshold(gr, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        _, img2 = cv2.threshold(gr, 200, 255, cv2.THRESH_BINARY)
        pshow(np.hstack([gr,img1,img2]))
        '''


def mean_HSV_color_of_masked_area(img, mask):
    import numpy as np
    pks = []
    hgt, wdt = mask.shape
    # Gjør om hver fiksels farge til HSVs hue verdi fra 0-360
    # Deretter beregne vi denne vinkelen om til det kartesiske kordinater.

    for i in range(hgt - 1):  # X koordinat
        for j in range(wdt - 1):  # Y koordinat
            if mask[i][j] == 255:
                pks.append(img[i][j][0])
    X = sum(np.cos(np.deg2rad(2 * pks)))/len(pks)
    Y = sum(np.sin(np.deg2rad(2 * pks)))/len(pks)
    mean = np.rad2deg(np.arctan2(Y, X))  # Gjennomsnittlig vinkel i grader

    # Tilslutt returneres vinkelen i OpenCV HSV, hue format, 0-179
    return mean / 2

def get_colors_from_image_Kmeans(img, number_of_colors):
    # Reshape img to line of pixels
    pixel_line = img.reshape(img.shape[1] * img.shape[0], 3)
    #
    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(pixel_line)
    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]  # Sortert etter størrelse
    rgb_colors = [ordered_colors[i] for i in counts.keys()]  # Sortert etter størrelse
    return rgb_colors

def show(img):
    cv2.imshow('Trykk_knapp_for_neste', img)
    cv2.waitKey()

def pshow(img):
    plt.imshow(img)
    plt.show()


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


def main(bilde_url,debug=False):
    if bilde_url == 'test':
        bilde_url = 'Bildebehandling/Bilder/Fotomosaikk/'
    snitt = []
    for t in range(15,16):
        try:
            t1 = time.perf_counter()
            print(t)
            # Variabler til lagring av sider
            top = None
            top_plassert = False
            kortsider = []
            langsider = []
            # Leser inn de fem bildene, Kjører bildebehandling, kategorisere sidetype: kort/lang/top
            for i in range(1,6):
                #side = Side(cv2.imread(bilde_url+f'bilde_{i}.jpg'), debug=False)
                side = Side(cv2.imread(bilde_url + f'SS{i}_ ({t}).jpg'), debug=False)
                side.find_box_and_crop_img()
                side.define_attributes()

                if side.type == 'top':
                    top = side
                elif side.type == 'kortside':
                    kortsider.append(side)
                else:
                    langsider.append(side)
            ### Finne rekkefølge på sider ###
            thresh = 3 # HSV fargetone terskelverdi som angir intervall for at farge skal regnes som lik.
            # 1. Velger en langside.
            rekkefolge = [langsider[0]]
            # 2. Finner kortside som har tilhørende fargekode
            ref_col = langsider[0].constraints['right']
            for side in kortsider:

                if thresh + side.constraints['left'] > ref_col >= side.constraints['left'] - thresh:
                    # 3. Skal denne langsiden ha topplokket over seg?
                    if thresh + langsider[0].constraints['top'] > top.constraints['bottom'] >= \
                            langsider[0].constraints['top'] - thresh:
                        rekkefolge.append(top)
                        top_plassert = True
                    # 4. Plasser korrekt kortside
                    rekkefolge.append(side)
                    kortsider.remove(side)
            # 5. Neste langside, legger toppen til her hvis den ikke blei lagt til tidligere
            rekkefolge.append(langsider[1])
            if not top_plassert:
                rekkefolge.append(top)
            # 6. Legger til siste kortside
            rekkefolge.append(kortsider[0])

            # 7. Vis frem bildet
            ferdig_bilde = draw_mosaic(rekkefolge)
            ferdig_bilde =  cv2.resize(ferdig_bilde, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            cv2.imwrite('Brukergrensesnitt/static/images/mosaikk.png',ferdig_bilde)
            #    cv2.imshow('Ferdig', ferdig_bilde)
            #    cv2.waitKey()
            tid = time.perf_counter() - t1
            snitt.append(tid)
            print(f'Nr: {round(tid,4)} s')

            img = cv2.cvtColor(ferdig_bilde, cv2.COLOR_BGR2RGB)
            if debug:
                pshow(img)
        except Exception as e:
            print("Oops!", e.__class__, "occurred in main function.")
            print(e.args)
            traceback.print_exc()
    #print(f'Snitt: {round((sum(snitt) ) / len(snitt),3)} [s]')


def draw_mosaic(sides):
    x = 0
    y = 200
    blank_image = np.zeros((800, 2600, 3), np.uint8)
    for i in range(len(sides)):
        if sides[i].type == 'top':
            x -= x_axis
            y = 0
        y_axis, x_axis, channels = sides[i].img_bgr.shape
        blank_image[y:y + y_axis, x:x + x_axis] = sides[i].img_bgr
        x += x_axis
        y = 200
    return blank_image[:400, :x]


def utv():
    for n in range(6):
        for i in range(5):
            #img = cv2.imread(f'Github/Eksperimentelt/Bilder/Konteiner/mate_{i+1}.png')
            img = cv2.imread(f'/Bildebehandling/Bilder/Fotomosaikk/SS{i + 1}_ ({n + 1}).jpg')
            con = Side(img, debug=True)
            #con.find_box_and_cut()
            con.test2()



if __name__ == '__main__':
    main('test')
    #utv()

'''
def find_box_and_crop_img(self):

        # 1. Utjevning av bilde
        img = cv2.medianBlur(self.img, 9)
        # img = self.img
        # hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # limits = HSV_colors.get('blå')

        # 2. Gråskala bilde
        ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # mask = cv2.inRange(hsv, limits[0], limits[1])

        # canny = cv2.Canny(ref_gray, 70, 180)
        # pshow(canny)

        # 3. Bruker Threshold funksjonen til å lage svart hvitt bilde.
        _, thresh = cv2.threshold(ref_gray, 200, 220, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        # 4. Finner konturene til objekter på sort hvitt bildet
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Finner den største konturen.
        largest_contour = max(contours, key=cv2.contourArea)
        contours.sort(key=cv2.contourArea, reverse=True)
        if self.show:
            empty = np.zeros_like(img)
            empty = cv2.drawContours(empty, [largest_contour], 0, (255, 255, 255), -1)  # Draw filled contour in mask
            pshow(empty)
            # for cnt in contours:
            #    empty = cv2.drawContours(empty, [cnt], 0, (100,200,71), -1)  # Draw filled contour in mask
            #    pshow(empty)

        # 6. Finner boksen som omkranser denne konturen
        # x, y, w, h = cv2.boundingRect(largest_contour)

        # 7. Klipper ut det som skal være bildet av konteineren.
        # self.img_bgr = self.img[y:y + h, x:x + w]

        rekt = cv2.minAreaRect(largest_contour)

        # 4.2 Gjør om formatet til "rect" til et "Contour" format
        boks = cv2.boxPoints(rekt)
        boks = np.int0(boks)

        boks = find_corner_points(boks, self.img.shape[0] / 2)
        self.forhold = img.shape[1] / self.img.shape[0]
        width = int(self.forhold * self.piksler_i_hoyden)
        self.img_bgr = change_perspective_and_crop(self.img, boks, width, self.piksler_i_hoyden)
'''
