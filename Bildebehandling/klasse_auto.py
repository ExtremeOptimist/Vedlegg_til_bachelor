import numpy as np
import cv2, time
from Bildebehandling.klasse_havbunn import show, pshow, rotation, translation

HSV_colors = {'blå': [np.array([105, 30, 30], dtype='uint8'), np.array([115, 255, 255], dtype='uint8')],
              'gul': [np.array([20, 100, 100], dtype='uint8'), np.array([30, 255, 255], dtype='uint8')],
              }

class Image:
    def __init__(self, img, debug=True):
        self.debug = debug
        self.img = img
        self.yellow_pip_min_areal = 750
        self.yellow_pip_max_areal = 5000
        self.yellow_pip_min_height = 15
        self.yellow_pip_max_height = 60
        self.yellow_pip_min_width = 100
        self.yellow_pip_max_width = 350
        self.max_solidity  = 0.95
        self.planteplass_min_areal = 2000

    def initial_image_transformations(self):
        f = 1 - (self.img.shape[0] - 500) / self.img.shape[0]
        self.img = cv2.resize(self.img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        self.hsv_median_blur_kernel = 5
        # Transforming from BGR to HSV format.
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # Blurring of hsv image.
        self.img_hsv = cv2.medianBlur(self.img_hsv, self.hsv_median_blur_kernel)

    def get_angle_and_translation(self):
        try:
            ## Leter etter blå linjer i bildet.
            ## 1 Henter inn definert farge
            limits = HSV_colors.get('blå')
            ## 2 Henter ut pixler som har definert farge og lager maskering
            mask = cv2.inRange(self.img_hsv, limits[0], limits[1])
            ## 3 Morfologisk Lukking til å koble sammen nærliggende objekter i lengderetningen.
            kernel = np.ones((20, 1), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=2)
            ## 4 Henter ut konturene og sortere dem etter areal
            contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours.sort(key=cv2.contourArea, reverse=True)  # Sortert i synkende rekkefølge etter areal.
            if len(contours) >= 2:
                ## DEBUG Tegner inn og viser de to største konturene
                if self.debug:
                    empty = np.zeros(self.img_hsv.shape[:2], dtype=np.uint8)  # Alternativt: emtpy = np.zeros_like(mask)
                    empty = cv2.drawContours(empty, [contours[0]], 0, color=255, thickness=-1)
                    show(empty)
                    empty = cv2.drawContours(empty, [contours[1]], 0, color=255, thickness=-1)
                    show(empty)

                ## 5 Henter data fra rør konturene.
                angles = []
                pos = []
                for i in range(0, 2):
                    [vx, vy, x, y] = cv2.fitLine(contours[i], cv2.DIST_HUBER, 0, 0.01, 0.01)
                    ang = np.rad2deg(np.arctan(vy[0] / vx[0]))
                    if ang > 0:
                        ang = ang - 180
                    angles.append(ang)
                    (c_x,c_y),storrelse,vinkel = cv2.minAreaRect(contours[i])
                    pos.append([c_x,c_y])

                ## DEBUG: Tegner inn divere informasjon og viser frem:
                    if self.debug:
                        print(f'kontur - {i}, vx:{vx}, vy:{vy}, x:{x}, y:{y}.')
                        # Tegne linjen inn på bildet.
                        rader, kolonner = self.img.shape[:2]  # Henter ut størrelsen på bildet
                        # Punktet hvor linjen passerer null på x-aksen.
                        venstre_y = int((-x * vy / vx) + y)
                        # punktet hvor linjen passerer x-aksen ut av bildet.
                        høyre_y = int(((kolonner - x) * vy / vx) + y)
                        # Tegner inn linja ved å definere to punkter.
                        #cv2.circle(self.img, (int(x),int(y)),10,(255,213,0),2)
                        cv2.circle(self.img, (int(c_x),int(c_y)), 10, (200, 120, 189), 4)
                        cv2.line(self.img, (kolonner - 1, høyre_y), (0, venstre_y), (0, 255, 0), 2)
                if self.debug:
                    show(mask)
                    show(self.img)

                    print(f' Vinkel med fitLines 180: 1 x {angles[0]}, 2 x {angles[1]}')

                ## 6 Beregner vinkel og forskyvning

                # Sortert senter kordinatene etter stigende X- verdi.
                pos.sort(key=lambda x: x[0])
                # Gjennomsnittsvinkelen til de to rørene, minus 90grader pga bildets rotasjon.
                angle = round(-90 - sum(angles)/2,2)
                # Negativ vinkel er "opp", motasatt av normalt aksesystem pga bildekordinatene hvor positiv y-akse er nedover.

                # Beregner avviket fra senter i antall piksler
                avvik_pos = (self.img.shape[1]/2) - (pos[0][0]+pos[1][0])/2
                # Beregner meter/piksel. Tar utgangspunkt i informasjon om at det er en meter fra blått rør til blått rør.
                avstand_per_piksel = 1 / (pos[1][0] - pos[0][0])
                # Gjør om posisjonsavviket til meter.
                avvik_fra_senter_i_meter = round(avvik_pos*avstand_per_piksel,5)
                # Negativ verdi = forskyvning ned i bildet i forhold til senter av bildet
            else:
                angle = 0
                avvik_fra_senter_i_meter = 0
            return angle, avvik_fra_senter_i_meter
        except:
            # Ved feil meldes avvik lik null
            return 0,0

    def find_end_bar(self):
        try:
            ### Leter etter gule rør

            ## 1 Henter ut pixler som har definert farge og lager maskering
            limits = HSV_colors.get('gul')
            mask = cv2.inRange(self.img_hsv, limits[0], limits[1])
            ## 2 Henter ut konturene og sortere dem etter areal
            contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours.sort(key=cv2.contourArea, reverse=True)  # Sortert i synkende rekkefølge et
            # 3 Vi luker ut planteplassene ved hjelp av Soliditet.

            if len(contours) >= 2:
                a1 = cv2.contourArea(contours[0])
                a2 = cv2.contourArea(contours[1])
                areal_convex_1 = cv2.contourArea(cv2.convexHull(contours[0]))
                areal_convex_2 = cv2.contourArea(cv2.convexHull(contours[1]))
                if areal_convex_1 > 0 and areal_convex_2 > 0 and a1 > self.planteplass_min_areal  and a2 > self.planteplass_min_areal:
                    solidity_1 = a1 / areal_convex_1
                    solidity_2 = a2 / areal_convex_2
                    if solidity_1 > self.max_solidity and solidity_2 > self.max_solidity:
                        contours.pop(0)
                        contours.pop(0)
                    elif solidity_1 > self.max_solidity:
                        contours.pop(0)
                    elif solidity_2 > self.max_solidity:
                        contours.pop(1)
            funn = False
            if len(contours) >= 2:
                ## 4 Henter ut informasjon om konturen
                x1, y1, b1, h1 = cv2.boundingRect(contours[0])
                a1 = cv2.contourArea(contours[0])
                x2, y2, b2, h2 = cv2.boundingRect(contours[1])
                a2 = cv2.contourArea(contours[1])
                senter = (y1+y2)/2 # Finner konurenes plassering i bildet. Har vi passert stoppstreken?
                ## 5 Sjekker om konturene fyller kravene vi har satt
                if (self.yellow_pip_max_height > h1 > self.yellow_pip_min_height and
                    self.yellow_pip_max_width > b1 > self.yellow_pip_min_width and
                    self.yellow_pip_max_areal > a1 > self.yellow_pip_min_areal) \
                        and (self.yellow_pip_max_height > h2 > self.yellow_pip_min_height and
                             self.yellow_pip_max_width > b2 > self.yellow_pip_min_width and
                             self.yellow_pip_max_areal > a2 > self.yellow_pip_min_areal):
                    funn = True
                    if self.debug:
                        cv2.rectangle(self.img, (x1, y1), (x1 + b1, y1 + h1), (255, 0, 50), 2)
                        cv2.rectangle(self.img, (x2, y2), (x2 + b2, y2 + h2), (255, 100, 50), 2)
                        show(mask)
                        show(self.img)
                    ## 6 Har vi passert streken?
                    if senter > self.img.shape[0]/2:
                        if self.debug: print("FUNN")
                        return False , funn
            if self.debug: print("INGENTING")
            return True , funn
        except Exception as e:
            print(e)
            if self.debug: print("EKSEPTION")
            return True, False

def test():
    tid= []
    mal = []
    nr = [2,4,5,6]
    for n in nr:#range(1,7):
        if n == 1: a=26
        elif n == 2: a=26
        elif n == 3: a=26
        res = [1.949, -2.862, -4.041, 2.825,0.772, 15.612]
        res1 = [0.0401, -0.0717, 0.0151, 0.1724, -0.1854, 0.1003]
        #img1 = cv2.imread(f'Bilder/Auto_kjoring/Treffsikkerhet.jpg')
        img1 = cv2.imread(f'Bilder/Auto_kjoring/eks_{n}.jpg')
        for i in range(-100,101):
            try:
                #img = cv2.imread(f'Bilder/Auto_kjoring/kjoring_{n} ({i}).jpg')
                #img = cv2.imread(f'Bilder/Auto_kjoring/eks_{4}.jpg')
                img = rotation(img1, i*0.1)
                start = time.perf_counter()
                #pshow(img)
                img = Image(img,debug=False)
                img.initial_image_transformations()
                ang, forsk = img.get_angle_and_translation()
                satus,funn = img.find_end_bar()
                stopp = time.perf_counter()
                #mal.append(ang-i*0.1-res[n-1])
                mal.append(forsk-res1[n-1])
               # print(f'Nr: {n}_{i}, Vinkel: {ang} [grader], Forskyvning: {forsk} [m], Stoppsignal: {satus}, Objekter;{funn}, Tidsbruk:{round(stopp-start,3)*1000}ms')
               # print(f'Nr: {n}_{i}, Vinkel: {ang} [grader], Forskyvning: {forsk} [m], Stoppsignal: {satus}, Objekter;{funn}, Tidsbruk:{round(((stopp-start)*1000),3)}ms')
                stans = "Ja"
                full = None
                if satus:
                    stans = "Nei"
                print(f'{i} & {ang-i*0.1} & {forsk} & {stans} & {round(((stopp-start)*1000))} & Ja \\\ \hline')
                tid.append(round(((stopp - start) * 1000)))
            except ValueError as e:
                print('e')
                print(f'Feilet nr: {i}')
    snitt = round(sum(tid)/len(tid),1)
    print(f"Snitt tid; {snitt}")

    import math
    import math as m
    import matplotlib.pyplot as plt
    import statistics as stats
    plt.scatter(range(0, len(mal)), mal)
    plt.show()
    mean = stats.mean(mal)
    var = stats.variance(mal)
    stdev = stats.stdev(mal)

    import pandas as pd
    df = pd.array(mal)

    print('Oppgave 9a: \n Mean: {0:.4f}, Varians: {1:.4f}, Stdev: {2:.4f}'.format(mean, var, stdev))


if __name__ == "__main__":
    test()


'''
Tidlig utgave av funksjon for automatisk kjøring.

def automatisk_kjoring(debug=False,se_etter_sluttstrek_tid=15):
    aktiv = True

    # Initialisere PID regulaturene
    pid_rotasjon = PID(Kp=0.03, Ki=0.01, Kd=0.00, setpoint=0, output_limits=(-1,1), sample_time=None)
    pid_forskyvning = PID(Kp=1, Ki=0.1, Kd=0.00, setpoint=0, output_limits=(-1,1), sample_time=None)


    # Havbunn kamera via OpenCV, initialisers fra url
    url = "rtsp://192.168.3.116/1:554/user=admin&password=&channel=1&stream=0.sdp?"
    cap = cv2.VideoCapture(url)

    # Starter time
    start = round(time.perf_counter(), 1)

    # Variabeler til teller og måleverdier
    teller = 0
    rotasjon = [0]
    forskyvning = [0]

    ## IIR FILTER
    alpha_r = 0.1
    alpha_f = 0.1

    paadrag = [0, 0, 0, 0, 0, 0, 0, 0]
    motor_control_data = {"motor_control_data": {
        "hvf": 1,
        "hhf": 1,
        "hvb": 1,
        "hhb": 1,
        "vvf": 1,
        "vhf": 1,
        "vvb": 1,
        "vhb": 1,
        "manip1": 1,
        "manip2": 1,
        "manip3": 1,
        "manip4": 1}}

    while aktiv:
        t1 = time.perf_counter()
        teller += 1

        # Les inn fra kamera
        ret, frame = cap.read() # "frame" er bildet.
           # frame = cv2.imread(f'Github/Eksperimentelt/Bilder/auto/test_ ({3}).jpg')

        # Henter målinger fra bildet.
        img = Image(img=frame, debug=debug)
        vnk, forsk = img.get_angle_and_translation()

        # Filtrere målingene med IIR filter
        rotasjon.append(vnk*alpha_r + (1 - alpha_r)*rotasjon[teller-1])
        forskyvning.append(forsk*alpha_f + (1 - alpha_f)*forskyvning[teller-1])

        # Beregner pådrag
        gir = pid_rotasjon(rotasjon[teller]) # X-akse høyre
        lateral = pid_forskyvning(forskyvning[teller]) # X- akse ventsre side

        # Sender pådragsverdier til ROV
        konverter_og_post_til_rov(lateral, gir, 50,motor_control_data, paadrag)

        # Ser etter sluttstreken
        if start-time.perf_counter() < se_etter_sluttstrek_tid:
            aktiv = img.find_start_stop()

        # Printer ut tidsbruk, vinkel og forskyvning
        if debug:
            print(f'Vinkel: {vnk} [grader], Forskyvning: {forsk} [m], Tidsbruk:{round(time.perf_counter()-start,3)}')


        def konverter_og_post_til_rov(venstre_x, hoyre_x, skalering,styring, paadrag):
            _ = 0
            venstre_x, venstre_y, hoyre_x, hoyre_y, paadrag_venstre, paadrag_hoyre, vinkel_venstre, vinkel_hoyre = konverter(venstre_x, _, hoyre_x, _)
            if venstre_y != 0 or venstre_x != 0:
                paadrag[0] = skalering * (paadrag_venstre * (-math.sin(vinkel_venstre) + math.cos(vinkel_venstre)))
                paadrag[1] = skalering * (paadrag_venstre * (-math.sin(vinkel_venstre) - math.cos(vinkel_venstre)))
                paadrag[2] = skalering * (paadrag_venstre * (math.sin(vinkel_venstre) - math.cos(vinkel_venstre)))
                paadrag[3] = skalering * (paadrag_venstre * (math.sin(vinkel_venstre) + math.cos(vinkel_venstre)))
            else:
                for i in range(0, 4):
                    paadrag[i] = 0

            if hoyre_y != 0:
                for i in range(4, 8):
                    paadrag[i] = skalering * -hoyre_y
            else:
                for i in range(4, 8):
                    paadrag[i] = 0

            if hoyre_x > 0:
                paadrag[0] += skalering * hoyre_x
                paadrag[2] += skalering * hoyre_x
            elif hoyre_x < 0:
                paadrag[1] += skalering * -hoyre_x
                paadrag[3] += skalering * -hoyre_x

            for i in range(0, 8):
                if paadrag[i] > skalering:
                    paadrag[i] = skalering
                elif paadrag[i] < -skalering:
                    paadrag[i] = -skalering

            for i in range(0, 8):
                paadrag[i] = int(math.floor(paadrag[i]))

            styring["motor_control_data"]["hvf"] = paadrag[0]
            styring["motor_control_data"]["hhf"] = paadrag[1]
            styring["motor_control_data"]["hhb"] = paadrag[2]
            styring["motor_control_data"]["hvb"] = paadrag[3]
            styring["motor_control_data"]["vvf"] = paadrag[4]
            styring["motor_control_data"]["vhf"] = paadrag[5]
            styring["motor_control_data"]["vhb"] = paadrag[6]
            styring["motor_control_data"]["vvb"] = paadrag[7]
'''
