import cv2
import numpy as np
import matplotlib.pyplot as plt


class ObjectTotal:
    def __init__(self, image):
        # Hva hvert korallrev trenger
        # im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # kan forandre til RGB
        self.image = image
        # assert self.image is not None  # check if image is loaded
        self.image_contour = None
        self.image_copy = self.image.copy()
        self.parts = []  # liste med deler av korallrevet
        self.f_parts = []  # filtrert liste
        self.with_bokses = None

    def set_before(self):
        for part_ in self.f_parts:
            part_.new = False
        return

    def get_bokses(self, templates, filter_threshold=0.1):
        # Tar inn en eller flere templates
        unused, gray = lilla_contour(self.image)
        for template_ in templates:
            tmplgray = cv2.cvtColor(template_.image, cv2.COLOR_BGR2GRAY)
            match = cv2.matchTemplate(gray, tmplgray, cv2.TM_CCOEFF_NORMED)
            above = np.where(match >= template_.matching_threshold)
            for (x, y) in zip(above[1], above[0]):
                # Lager et del objekt for hver match over terskelen
                self.parts.append(ObjectPart(x, y, x+template_.width, y+template_.height,
                                             match[y, x], (0, 0, 0), True))
        filtered_parts = filter_overlap(self.parts, filter_threshold)
        for f_part_ in filtered_parts:
            self.f_parts.append(f_part_)
        return filtered_parts

    def draw_rectangles(self, parts=None):
        image = self.image.copy()
        colors = []
        if parts is None:
            parts = self.f_parts
        for part_ in parts:
            cv2.rectangle(
                image,
                (part_.ulx, part_.uly),
                (part_.brx, part_.bry),
                part_.col, 2)
        self.with_bokses = image
        return image
    
    def show(self, with_bokses=False, kontur=False):
        # plt.figure()
        if with_bokses:
            plt.imshow(self.with_bokses[:, :, ::-1])
        elif kontur:
            unused, gray = lilla_contour(self.image)
            plt.imshow(gray)
        else:
            plt.imshow(self.image[:, :, ::-1])
        # plt.show()

    def get_contour(self):
        # Returnerer lilla contur
        unused, kontur = lilla_contour(self.image)
        return kontur


class Template:
    def __init__(self, image, matching_threshold):
        self.image = image
        self.matching_threshold = matching_threshold
        self.height, self.width, self.depth = self.image.shape

    def show(self):
        # plt.figure()
        plt.imshow(self.image)


def make_template(path, similarity_thresh):
    template = cv2.imread(path)
    template = Template(template, similarity_thresh)
    return template


class ObjectPart:
    __id = 0  # unik id for hver del, gjør det lettere å se hva som forsvinner/står igjen.
    # Hva hver del av korallrevet trenger

    def __init__(self, upper_left_x, upper_left_y,
                 bottom_right_x, bottom_right_y, matching_value, color, new):
        self.ulx = upper_left_x
        self.uly = upper_left_y
        self.brx = bottom_right_x
        self.bry = bottom_right_y
        self.mav = matching_value
        self.col = color
        self.col_txt = ''
        self.new = new
        self.blc = False
        self.hld = False
        ObjectPart.__id += 1
        self._id = ObjectPart.__id

    def get_rectangle(self):
        # Gir ut liste med [x1, y1, x2, y2] koordinater
        # altså, hjørnene oppe venstre og nede høyre
        return [self.ulx, self.uly, self.brx, self.bry]

    def set_color(self):
        if self.new:  # new yes
            self.col, self.col_txt = (0, 255, 0), 'Ny'  # growth, green
        else:  # new no
            self.col, self.col_txt = (0, 255, 255), 'Død'  # death, yellow
        if self.hld:  #
            self.col, self.col_txt = (255, 0, 0), 'Helb.'  # recover, blue
        if self.blc:  # wht yes
            self.col, self.col_txt = (0, 0, 255), 'Blek.'  # bleaching, red
        return True

    def get_color_txt(self):
        if self.col == (0, 255, 0):
            text = 'green'

    def __str__(self):
        return f'id er {self._id}'

    def __repr__(self):
        return self.__str__()


def lilla_contour(image, hue=(155, 179)):

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # plt.figure()
    # plt.imshow(img_hsv)
    # color_low = np.array([hue[0], 30, 50]) # bra for mob
    # color_high = np.array([hue[1], 220, 255]) # bra for mob
    color_low = np.array([hue[0], 60, 90])  # bra for MATE
    color_high = np.array([hue[1], 200, 221])  # bra for MATE
    mask = cv2.inRange(img_hsv, color_low, color_high)
    res = cv2.bitwise_and(image, image, mask=mask)
    bare_h_lag = res[:, :, 0]
    # kan jeg ungå å bruke terskling?
    #ret, contour_med_stoy = cv2.threshold(bare_h_lag, 30, 250, cv2.THRESH_BINARY)
    contour_med_stoy = bare_h_lag

    contours, hir = cv2.findContours(contour_med_stoy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntssorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    cnt1 = cntssorted[:5]  # Tar de 6 første konturene, som skal være de seks største
    blank = np.zeros(bare_h_lag.shape)
    only_contour = cv2.drawContours(blank, cnt1, -1, 255, -1)
    only_contour = np.uint8(only_contour)
    return contour_med_stoy, only_contour


def remove_noise(image):
    contour_med_stoy = image
    blur = cv2.medianBlur(contour_med_stoy, 17)
    blob_size = 5
    kernel = np.ones((blob_size, blob_size), np.uint8)
    eroded = cv2.erode(blur, kernel, iterations=5)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    eroded2 = cv2.erode(dilated, kernel, iterations=3)
    return eroded2


def biggest_contour_roi(gray_img):
    # plt.figure()
    # plt.imshow(gray_img)
    # plt.show()
    contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_cont = max(contours, key=cv2.contourArea)
    bounding_rect = cv2.boundingRect(biggest_cont)
    return bounding_rect, biggest_cont, contours


def crop_image(image, rectangle_coords):
    sx = int(rectangle_coords[0] - (0.1 * image.shape[0]))
    if sx < 0:
        sx = 0
    sy = rectangle_coords[1]
    if sy < 0:
        sy = 0
    wd = int(rectangle_coords[2] + (0.2 * image.shape[0]))
    he = rectangle_coords[3]
    return image[sy:sy + he, sx:sx + wd]


def compute_iou(a, b):
    # Tar inn to rektangler,
    # a, b = [x1, y1, x2, y2]
    # og regner ut prosentvis overlapp
    # 0 = 0 % 0.5 = 50 % 1 = 100 %
    epsilon = 1e-5

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    width = (x2 - x1)  # Area of intersection
    height = (y2 - y1)

    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def filter_overlap(list_of_parts, threshold):
    # Filtrerer bokser som overlapper over en terskel
    non_overlapping = []
    for obj in list_of_parts:
        overlap = False
        for f_obj in non_overlapping:
            iou = compute_iou(obj.get_rectangle(), f_obj.get_rectangle())  # se funskjon "compute iou"
            if iou > threshold:
                overlap = True
                break
        if not overlap:
            non_overlapping.append(obj)
    return non_overlapping


def find_non_overlap(objects_before, objects_after, non_max_suppression_threshold=0.05):
    filtered_objects = []
    if len(objects_before) <= len(objects_after):
        o_list_1 = objects_after
        o_list_2 = objects_before
    else:
        o_list_1 = objects_before
        o_list_2 = objects_after
    combined = objects_before + objects_after
    for o1 in combined:  # Går gjennom alle objekter
        overlap_found = False  # Sier at man finner overlapp med en gang
        for o2 in o_list_1:  # sjekker med alle objekter i liste 2
            iou = compute_iou(o1.get_rectangle(), o2.get_rectangle())
            # Regner ut iou for object og objekt som ikke har overlapp
            if iou >= non_max_suppression_threshold:  # Hvis område er høyt betyr det at det er overlapp
                overlap_found = True
                break  # går ut av løkken hvis den finner overlapp
        if not overlap_found:  # Hvis ingen hadde område overlapp, vil øverste loop objekt gå til filtrert listen
            filtered_objects.append(o1)
    for o1 in combined:  # Går gjennom alle objekter
        overlap_found = False  # Sier at man finner overlapp med en gang
        for o2 in o_list_2:  # sjekker med alle objekter i liste 2
            iou = compute_iou(o1.get_rectangle(), o2.get_rectangle())
            # Regner ut iou for object og objekt som ikke har overlapp
            if iou >= non_max_suppression_threshold:  # Hvis område er høyt betyr det at det er overlapp
                overlap_found = True
                break  # går ut av løkken hvis den finner overlapp
        if not overlap_found:  # Hvis ingen hadde område overlapp, vil øverste loop objekt gå til filtrert listen
            filtered_objects.append(o1)
    return filtered_objects


def is_area_white(roi):
    # finner lilla, og gjør om til svart
    unused, kontur = lilla_contour(roi)
    general_pixel_intensity = cv2.mean(kontur)
    if general_pixel_intensity[0] < 15:
        result = True
    else:
        result = False
    return result


def check_white_parts(im_bef, im_aft, list_of_parts):
    if list_of_parts is not []:
        for part_ in list_of_parts:
            roi1 = im_bef[part_.uly:part_.bry, part_.ulx:part_.brx, :]
            roi2 = im_aft[part_.uly:part_.bry, part_.ulx:part_.brx, :]
            a = is_area_white(roi1)
            b = is_area_white(roi2)
            if a and not b:
                part_.hld = True
            if b and not a:
                part_.blc = True
    return True


def prep_image(path, shape):
    # Tilpass bilde før. (kan gjøres til funksjon)
    im = cv2.imread(path)
    kontur, unused = lilla_contour(im)  # Bare lilla områder
    roi_coords, biggest_cont_, counts = biggest_contour_roi(kontur)  # Korallrev bounding box
    roi = crop_image(im, roi_coords)  # klippet ut
    prep_img = cv2.resize(roi, shape, interpolation=cv2.INTER_AREA)
    return prep_img


def extract_color(image, hue=(121, 177), crop=True):
    blob_size = 5
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_low = np.array([hue[0], 60, 50])
    color_high = np.array([hue[1], 200, 221])
    mask = cv2.inRange(img_hsv, color_low, color_high)  # lager maske
    res = cv2.bitwise_and(image, image, mask=mask)  # Bildet som representerer fargen i korallrevet
    # koden over, bitwise and brukes for å sortere ut de lilla pikslene som vi er interreserte i.
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)  # Gjør til BGR
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  # Gjør til gråskala
    kernel = np.ones((blob_size, blob_size), np.uint8)
    retur_img = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    ret, res = cv2.threshold(retur_img, 90, 250, cv2.THRESH_BINARY)
    contours, hir = cv2.findContours(res, 1, 2)
    biggest_cont = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest_cont)
    if crop:
        img_cropped = image[y:y + h, x:x + w]
        contour_cropped = res[y:y + h, x:x + w]
    else:
        img_cropped = image
        contour_cropped = res
    return contour_cropped, img_cropped


def crop_filename(text):
    text = str.split(text, '/')
    text = text[4]
    text = str.split(text, '.')
    text = text[0]
    text_no_ul = text.replace('_', '\\_')
    return text, text_no_ul
