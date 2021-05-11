import cv2
import matplotlib.pyplot as plt
import Bildebehandling.tools as tools
import os


def find_differences(im1, im2):
    # Tilpass bilde før. (kan gjøres til funksjon)
    prep_img_for = tools.prep_image(im1, shape=(400, 400))

    # Tilpass bildet etter
    prep_img_etr = tools.prep_image(im2, shape=(400, 400))

    # Kode for å sette opp malene
    path_templates = ['Bachelor-2021/Vedlegg_til_bachelor/Bildebehandling/templates/MT_1.jpg',
                      './Bachelor-2021/Vedlegg_til_bachelor/Bildebehandling/templates/MT_2.jpg',
                      './Bachelor-2021/Vedlegg_til_bachelor/Bildebehandling/templates/MT_3.jpg']
    thresh_templates = [0.95, 0.7, 0.7]

    templates = [
        tools.make_template(path_templates[0], thresh_templates[0]),
        tools.make_template(path_templates[1], thresh_templates[1]),
        tools.make_template(path_templates[2], thresh_templates[2])
        # tools.make_template('./templates/EG_4.jpg', 0.95)
    ]

    # Kode for å hente ut bokser før
    kfor = tools.ObjectTotal(prep_img_for)
    bfor = kfor.get_bokses(templates, filter_threshold=0.1)

    # Kode for å hente ut bokser etter
    ketr = tools.ObjectTotal(prep_img_etr)
    betr = ketr.get_bokses(templates, filter_threshold=0.1)

    # Finner forskjeller
    diff_bokses = tools.find_non_overlap(bfor, betr)
    kfor.set_before()  # setter bokser fra bilde før til gamle.

    # Sjekker for hvite deler, og setter riktig farge på bokser
    tools.check_white_parts(prep_img_for, prep_img_etr, diff_bokses)
    colors = ''
    for part in diff_bokses:
        part.set_color()
        colors += (part.col_txt + ', ')

    # Tegner rektanglene med riktige farger på bilde etter
    diff_etr = ketr.draw_rectangles(diff_bokses)
    return diff_etr

