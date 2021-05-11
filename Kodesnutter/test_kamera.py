import cv2
import time


def ta_bilde(cam):
    try:
        ret, frame = cam.read()  # Henter bilde fra kamera
        cv2.imwrite('testbilde.jpg', frame)  # Lagrer bildet til jpg fil.
        return True
    except:
        return False

if __name__ == '__main__':
    url_front = "rtsp://192.168.3.116/1:554/user=admin&password=&channel=1&stream=0.sdp?"
    FRONT_CAM = cv2.VideoCapture(0) # 0-Webkamerea , eller Url til IP-kamera
    vellykket = []
    feilet = []

    for i in range(10):
        start = time.perf_counter()
        if ta_bilde(FRONT_CAM):
            vellykket.append(time.perf_counter() - start)
        else:
            feilet.append(time.perf_counter() - start)

    gj_snitt = round(sum(vellykket) / len(vellykket), 3) * 1000  # [ms]
    print(f'Antall vellyket: {len(vellykket)}, '
          f'Gjennomsnittstid vellykeket resultat {gj_snitt},'
          f'Antallfeilet: {len(feilet)}, ')
