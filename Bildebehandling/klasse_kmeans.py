"""Skript for å kjøre løsning to af bildebehandling.
Har nøkkelpunkter, vil transformere korallrevene til å matche,
Deretter bruke SSIM for å sammenligne ujevnheter. Laget av Bjørnar"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import graph_tools as gt
import tools
# kmeanstest

SHAPE = (400, 400)  # Hvor mange pixler korralrevet skal skaleres ned til.
# shape er bra med 200, 200, men litt tregt
PDIST = 6  # distance mellom punkter
NR_CLUSTERS = 150  # 140 virker bra med 400 x 400
# ANGLE_TOLERANCE = 40  # Slingrigsmunn i grader
path_for = './Bjornar_python/ref_images/EG_mate_for.jpg'
path_etr = './Bjornar_python/ref_images/EG_fire_to.jpg'

# k-means
# Basert på scikit learn, maskinlæring algoritmer
# med kile : https://realpython.com/k-means-clustering-python/

# standardisering:
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(all_pixels)

# ================================================
# Tre klasse som egentlig skal bestå av linjestykkene,
# men foreløpig alle punktene. Kan og ha punkter og linjestykker
# Etterhvert


def comp_branches_length(obj1, obj2, thre):
    diffs = []
    for bfor in obj1:
        cfor = bfor.category
        for betr in obj2:
            cetr = betr.category
            if cfor == cetr:
                lfor = bfor.get_length()
                letr = betr.get_length()
                diff = lfor - letr
                if thre < diff:
                    diffs.append(bfor)
                    diffs.append(betr)
                    bfor.diff = diff
                    betr.diff = diff
                    # fordi den adder den siste, som er den positive,
                    # vil ikke ta med negative verdier
    return diffs


def compare_branches(bf, be):
    # tar inn grener, og returnerer de som ikke er i begge listene.
    # setter og alder, 'new' for gren etter 'old' for før.
    # og setter verdien til branch.diff true dersom de er forskjellige
    bfcs = []
    becs = []
    for branch in bf:
        branch.age = 'old'
        bfc = branch.category
        bfcs.append(bfc)
    for branch in be:
        branch.age = 'new'
        bec = branch.category
        becs.append(bec)
    bfcset = set(bfcs)
    becset = set(becs)
    diffs = bfcset ^ becset  # elementer som er i en eller to, men ikke begge
    for branch in (bf + be):
        cat = branch.category
        if cat in diffs:
            branch.diff = True
    # diff_ = comp_branches_length(bf_, be_, thre=20)
    return diffs


def get_braches(img_path):
    # Leser inn bilde:
    prep_img = tools.prep_image(img_path, shape=SHAPE)
    unused, img = tools.lilla_contour(prep_img)
    plt.imshow(img)
    plt.title(img_path)
    plt.show()
    img_points = gt.pixels_to_datapoints(img)
    img_k_centers = gt.get_kmeans_centers(img_points, NR_CLUSTERS)
    plt.scatter(img_k_centers[:, 0], img_k_centers[:, 1])
    # Med filtrering
    plt.show()
    verticies, mst, k_centers = gt.get_graph(img_k_centers)
    # uten filtrering
    # verticies, mst, k_centers = gt.get_graph_no_filter(img_k_centers)
    plt.figure()
    gt.plot_mst(mst, k_centers, 'før')
    # three_neighbor_verticies = gt.find_all_threes_verticies(v_for)
    plt.show()
    gt.get_keypoints(verticies, k_centers)  # pass på å bruke riktige punkter
    branches = gt.make_branches(verticies)
    gt.categorize_branches(branches, verticies)
    return branches


def draw_differences(branches):
    # uferdig
    for branch in branches:
        branch.draw_box()
    pass


if __name__ == '__main__':
    matplotlib.use('qt5agg')

    bf_ = get_braches(path_for)
    be_ = get_braches(path_etr)
    print('Grener før:\n')
    for branch_ in bf_:
        print(branch_.category)
    print('\nGrener etter:\n')
    for branch_ in be_:
        print(branch_.category)
    #
    # For å lagre punktene, for raskere testing
    # np.save('bfc', bilde_for_clusters, False)
    # np.save('bec', bilde_etter_clusters, False)
    #
    # må nå sammenligne grennene
    # plaenen er å gå gjennom grener for, grener etter i nestet løkke
    # når den treffer på samme kategori, sammelnigner lengdene
    # dersom lengdenen avviker mer enn en terskel, sjekk
    diff_ = compare_branches(bf_, be_)
    print('\nDifferanser:\n')
    print(diff_)
    # #
    #
    #
    # dersom differansen er negativ, betyr det at etter er lengre enn før.
    #
    # Plotting
    # plt.subplot(121)
    # gt.plot_mst(mst_for2, pnts_for2, 'før')
    # plt.subplot(122)
    # gt.plot_mst(mst_et2, pnts_et2, 'etter')
    #
    plt.show()

# For plotting av bilde kontur osv.
    # plt.subplot(121)
    # plt.imshow(bilde_for)
    # bilde_for = remove_noise(bilde_for)
    # plt.subplot(121)
    # plt.imshow(bilde_for)
    # bilde_for = extract_contour(bilde_for)
    # plt.subplot(121)
    # plt.imshow(bilde_for)
    #
    # plt.subplot(121)
    # plt.scatter(pnts_for2[:, 0], pnts_for2[:, 1])
    # plt.subplot(122)
    # plt.imshow(bilde_for)