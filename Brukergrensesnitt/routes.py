from Brukergrensesnitt import app, bcrypt, login_manager
from Brukergrensesnitt.forms import LoginForm
from Brukergrensesnitt.models import User
from flask import render_template, flash, redirect, url_for, request
from flask_login import login_user, logout_user, current_user, login_required
import datetime as dt
import json, time
import cv2 ,random, traceback
from Bildebehandling.klasse_auto import Image
from Bildebehandling.klasse_havbunn import Seabed
from Bildebehandling.klasse_konteiner import main
import Bildebehandling.klasse_korallrev as korallrev
### JSON-filer til kommunikasjon mellom ROV og brukergrensesnitt.
status_data = {"status_data": {
    "teller":  0,
    "tid":  0,
    "manuel": 1,
    "start": 1,
    "lys": 1,
    "mROV": 1,
    "manip_paa": 0,
    "depth_regulator": 0,
    "stamp_regulator": 1,
    "rull_regulator": 1,
    "lekasje": 0,
    "dybde": 102,
    "hoyde": 0,
    "temp_vann": 0,
    "temp": 21,
    "aks_x": 1.21,
    "aks_y": 12.21,
    "aks_z": 12,
    "vinkel_x": 21,
    "vinkel_y": 54,
    "vinkel_z": 65,
    "strom_1": 1,
    "strom_2": 0.3,
    "strom_3": 0.43,
    "spenn_1": 0.432,
    "spenn_2": 0.432,
    "spenn_3": 0.432,
    "hvf": 32,
    "hhf": -21,
    "hvb": 44,
    "hhb": 87,
    "vvf": -32,
    "vhf": 43,
    "vvb": -90,
    "vhb": -75,
    "manip1": 33,
    "manip2": -22,
    "manip3": -59,
    "manip4": -5
}}
control_data = {"control_data" : {
    "venstre_x": 14,
    "venstre_y": 15,
    "hoyre_x": 81,
    "hoyre_y": 71,
    "manip1": 12,
    "manip2": -22,
    "manip3": -59,
    "manip4": -5,
    "skalering": 50,
    "manuel": 1,
    "mROV": 0,
    "manip_paa": 0,
    "depth_regulator": 0,
    "stamp_regulator": 1,
    "rull_regulator": 1,
    "lys": 1
}}
control_data_mROV = {"control_data_mROV":{
    "stikke1_x": 12,
    "stikke1_y": 13,
    "stikke2_y": 41,
    "mROV": 0,
    "lys": 0,
    "bryter_ext": 0
}}
status_data_mROV = {"status_data_mROV":{
    "temp": 23,
    "bryter_ext": 0,
    "mROV": 0,
    "lys": 0,
    "strom_1": 1,
    "strom_2": 0.3,
    "strom_3": 0.4,
    "strom_4": 1.1,
    "motor_vertikal_1": 59,
    "motor_vertikal_2": -90,
    "motor_horisontal_1": 32
}}
regulator = {"regulator" : {
    "KP_depth": 0.125,
    "KP_stamp": 0.99,
    "KP_rull": 1.44,
    "KI_depth": 0.25,
    "KI_stamp": 0.56,
    "KI_rull": 4.22,
    "KD_depth": 0.5,
    "KD_stamp": 0.24,
    "KD_rull": 5.11
}}
status_regulator = { "status_regulator" : {
    "KP_depth": 0.125,
    "KP_stamp": 0.99,
    "KP_rull": 1.44,
    "KI_depth": 0.25,
    "KI_stamp": 0.56,
    "KI_rull": 4.22,
    "KD_depth": 0.5,
    "KD_stamp": 0.24,
    "KD_rull": 5.11
}}

FRONT_CAM = None
mROV_CAM = None
BUNN_CAM = None
BILDE_TELLER = 1
auto_not_initialisert = 1
vente = True
liste = []
teller = 1
AUTO_TELLER = 0
tid = 0

start = time.perf_counter()
status_data['status_data']['tid'] = start


@app.route('/',methods=["GET", "POST"])
@app.route('/gui',methods=["GET", "POST"])
@login_required
def home():
    return render_template('ROV.html', title="Hymir")


@login_required
@app.route("/ta_bilde_havbunnKamera")
def ta_bilde_havbunnKamera():
    try:
        global BILDE_TELLER   # Ønsker globale variabler
        global BUNN_CAM
        if BUNN_CAM is None or BUNN_CAM.isOpened() is not True:
            url1 = "rtsp://192.168.3.116/1:554/user=admin&password=&channel=1&stream=0.sdp?"
            BUNN_CAM = cv2.VideoCapture(0)
        if BUNN_CAM.isOpened():
            _, bilde = BUNN_CAM.read()  # "frame" er bildet.
            if cv2.imwrite(f'Brukergrensesnitt/static/images/nr_{BILDE_TELLER}.png',bilde):
                BILDE_TELLER += 1
                return render_template('bilde.html', nr=str(BILDE_TELLER-1))
    except Exception as e:
        print("Oops!", e.__class__, "occurred in TA bilde.")
        print(e.args)
        return ' Feilet'


@login_required
@app.route("/ta_bilde_frontkamera")
def ta_bilde_frontkamera():
    try:
        global BILDE_TELLER  # Ønsker globale variabler
        global FRONT_CAM
        if FRONT_CAM is None or FRONT_CAM.isOpened() is not True:
            url1 = "rtsp://192.168.3.116/1:554/user=admin&password=&channel=1&stream=0.sdp?"
            FRONT_CAM = cv2.VideoCapture(0)
        if FRONT_CAM.isOpened():
            _, bilde = FRONT_CAM.read()
            if cv2.imwrite(f'Brukergrensesnitt/static/images/nr_{BILDE_TELLER}.png', bilde):
                BILDE_TELLER += 1
                return render_template('bilde.html', nr=str(BILDE_TELLER-1))
    except Exception as e:
        print("Oops!", e.__class__, "occurred in TA bilde.")
        print(e.args)
        return ' Feilet'


@login_required
@app.route("/ta_bilde_mROV")
def ta_bilde_mROV():
    try:
        global BILDE_TELLER  # Ønsker globale variabler
        global mROV_CAM
        if mROV_CAM is None or mROV_CAM.isOpened() is not True:
            url1 = "rtsp://192.168.3.116/1:554/user=admin&password=&channel=1&stream=0.sdp?"
            mROV_CAM = cv2.VideoCapture(0)
        if mROV_CAM.isOpened():
            _, bilde = mROV_CAM.read()
            if cv2.imwrite(f'Brukergrensesnitt/static/images/nr_{BILDE_TELLER}.png', bilde):
                BILDE_TELLER += 1
                return render_template('bilde.html', nr=str(BILDE_TELLER-1))
    except Exception as e:
        print("Oops!", e.__class__, "occurred in TA bilde.")
        print(e.args)
        return ' Feilet'


@login_required
@app.route("/slett_siste_bilde")
def slett_siste_bilde():
    global BILDE_TELLER  # Ønsker globale variabler
    BILDE_TELLER -= 1
    return render_template('bilde.html', nr=str(BILDE_TELLER))


@login_required
@app.route("/fotomosaikk")
def fotomosaikk():
    main('test',debug=False)
    return render_template('mosaikk.html')


@login_required
@app.route("/havbunn")
def havbunn():
    ##img = cv2.imread(f'Brukergrensesnitt/static/images/nr_{BILDE_TELLER}.png')
    img = cv2.imread(f'Bildebehandling/Bilder/Oversikt_havbunn/Oversiktsbilder/test_ ({5}).jpg')
    map = cv2.imread('Bildebehandling/Bilder/Oversikt_havbunn/havbunn.png')
    map = cv2.rotate(map, cv2.ROTATE_90_CLOCKWISE)
    a = Seabed(img, map, debug = False, debug_square = False, inspection = False)
    a.initial_image_transformations()
    a.find_roi()
    a.split_roi_into_squares()
    resultat = cv2.cvtColor(a.draw_map(), cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'Brukergrensesnitt/static/images/map.png', resultat)
    return render_template('havbunn.html')


@login_required
@app.route("/korall")
def koralrev():
    path_img_bef = f'Brukergrensesnitt/static/images/korallrev_for_mate.png'
    path_img_aft = f'Brukergrensesnitt/static/images/nr_{BILDE_TELLER}.png'
    resultat = korallrev.find_differences(path_img_bef, path_img_aft)
    cv2.imwrite(f'Brukergrensesnitt/static/images/map.png', resultat)
    return render_template('korall.html')


@login_required
@app.route("/init_kamera")
def init_kamera():
    # Initialiser kamera på ROV og mROV
    global FRONT_CAM
    global BUNN_CAM
    global mROV_CAM
    global BILDE_TELLER
    global url_front
    global url_bunn
    global url_mrov
    try:
        url_front = "rtsp://192.168.3.116/1:554/user=admin&password=&channel=1&stream=0.sdp?"
        FRONT_CAM = cv2.VideoCapture(0)
        url_bunn = "rtsp://192.168.3.117/1:554/user=admin&password=&channel=1&stream=0.sdp?"
        BUNN_CAM = cv2.VideoCapture(0)
        url_mrov = "rtsp://192.168.3.118/1:554/user=admin&password=&channel=1&stream=0.sdp?"
        mROV_CAM = cv2.VideoCapture(0)
        BILDE_TELLER = 0
    except:
        return "feilet"
    return "suksee"

@login_required
@app.route("/del_kamera")
def del_kamera():
    try:
        FRONT_CAM.release()
        BUNN_CAM.release()
        mROV_CAM.release()
    except:
        return 0
    return 1

@login_required
@app.route("/auto")
def auto():
    try:
        global auto_not_initialisert

        if auto_not_initialisert:
            auto_not_initialisert = initialiser_auto()

        global BUNN_CAM
        global AUTO_TELLER
        AUTO_TELLER += 1

        # Les inn fra kamera
        if BUNN_CAM == None or BUNN_CAM.isOpened() != True:
            BUNN_CAM = cv2.VideoCapture(0)
            ret, frame = BUNN_CAM.read()
        else:
            ret, frame = BUNN_CAM.read()
        frame = cv2.imread(f'Bildebehandling/Bilder/Auto_kjoring/test_ ({3}).jpg')

        # Henter målinger fra bildet.
        img = Image(img = frame, debug = False)
        vnk, forsk = img.get_angle_and_translation()

        # Filtrere målingene med IIR filter
        rotasjon.append(vnk * alpha_r + (1 - alpha_r) * rotasjon[AUTO_TELLER - 1])
        forskyvning.append(forsk * alpha_f + (1 - alpha_f) * forskyvning[AUTO_TELLER - 1])

        # Beregner pådrag
        control_data["control_data"]["venstre_x"] = pid_forskyvning(forskyvning[AUTO_TELLER])  #Ønsket svai bevegelse.
        control_data["control_data"]["hoyre_x"] = pid_rotasjon(rotasjon[AUTO_TELLER])  #Ønsket gir rotasjon.

        # Ser etter sluttstreken.
        if time.perf_counter() - start > se_etter_sluttstrek_tid:
            control_data["control_data"]["manuel"] = img.find_end_bar()
            control_data["control_data"]["manuel"] = 1
            auto_not_initialisert = 1
        # Skriv data til ROV
        update_data()
    except Exception as e:
        print("Oops!", e.__class__, "occurred in Auto.")
        print(e.args)
        traceback.print_exc()

        return status_data
    return status_data

@login_required
@app.route("/get_status_data.json")
def get_status_data():
    global teller
    global start
    global liste
    global vente
    global tid


    teller = teller + 1
    status_data['status_data']['teller'] = teller

    ##tid = round(time.perf_counter() - start,4)

    tid = 0.001 * round(time.perf_counter() - start,4) + 0.999* tid
    start = time.perf_counter()

    '''
        liste.append(tid)

    if len(liste) > 6000 and vente:
        liste.pop(0)
        liste.pop(0)
        liste.pop(0)
        print(sum(liste)/len(liste))
        print(min(liste))
        print(max(liste))
        x = range(len(liste))
        plt.hist(liste,bins=30)
        plt.show()
        vente = False
    '''
    #status_data['status_data']['teller'] = teller
    status_data['status_data']['tid'] = tid
    status_data["status_data"]["hvf"] = random.randint(-100,100)
    status_data["status_data"]["hhf"] = random.randint(-100,100)
    status_data["status_data"]["hhb"] = random.randint(-100,100)
    status_data["status_data"]["vvf"] = random.randint(-100,100)
    start = time.perf_counter()

    return status_data


@app.route("/post_control_data.json",methods=["POST"])
def post_control_data():
    inn_data = request.form
    control_data['control_data'].update(inn_data)
    status_data['status_data'].update(control_data['control_data'])
    return status_data


@app.route("/get_status_data_mROV.json")
def get_status_data_mROV():
    #global teller
    #teller = teller + 1
    status_data_mROV['status_data_mROV']['motor_horisontal_1'] = random.randint(-100,100)
    status_data_mROV['status_data_mROV']['motor_vertikal_2'] = random.randint(-100,100)
    status_data_mROV['status_data_mROV']['motor_vertikal_1'] = random.randint(-100,100)
    return status_data_mROV


@app.route("/post_control_data_mROV.json",methods=["POST"])
def post_control_data_mROV():
    inn_data = request.form
    control_data_mROV['control_data_mROV'].update(inn_data)
    status_data_mROV['status_data_mROV'].update(control_data_mROV['control_data_mROV'])
    update_data()
    return status_data_mROV


# Login page
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))   # If logged in, redirect to Home-page.
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(name=form.name.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, form.remember.data, duration=dt.timedelta(days=30))
            flash("Du er nå innlogget", "success")
            next_page = request.args.get("next")
            return redirect(next_page) if next_page else redirect(url_for("home"))
        else:
            flash("Ugyldig informasjon, kontroller brukernavn og passord!", "danger")
            return render_template("login.html", title="Logg inn", form=form)
    return render_template("login.html", title="Logg inn", form=form)


# This route does not have its own page. Put is used to preform log out operation
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


# How login manager selects user - THis could be don with changing-random-id- numbers, To minimize risk of using old session
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def update_data():
    global teller
    teller += 1
    status_data["status_data"].update(control_data["control_data"])

    status_data['status_data']['strom_1'] = teller
    status_data['status_data']['strom_2'] = teller
    status_data["status_data"]["manip1"] = random.randint(-100,100)
    status_data["status_data"]["manip2"] = random.randint(-100,100)
    status_data["status_data"]["manip3"] = random.randint(-100,100)
    status_data["status_data"]["manip4"] = random.randint(-100,100)
    return status_data

def initialiser_auto():
    try:
        from simple_pid import PID
        global pid_rotasjon
        global pid_forskyvning
        global AUTO_TELLER
        global start
        global se_etter_sluttstrek_tid
        global rotasjon
        global forskyvning
        global alpha_r
        global alpha_f
        global BUNN_CAM

        pid_rotasjon = PID(Kp = 0.03, Ki = 0.01, Kd = 0.00, setpoint = 0, output_limits = (-1, 1), sample_time = None)
        pid_forskyvning = PID(Kp = 1, Ki = 0.1, Kd = 0.00, setpoint = 0, output_limits = (-1, 1), sample_time = None)

        # Starter time
        start = time.perf_counter()
        se_etter_sluttstrek_tid = 15

        # Variabeler til teller og måleverdier
        AUTO_TELLER = 0
        rotasjon = [0]
        forskyvning = [0]

        ## IIR FILTER
        alpha_r = 0.1
        alpha_f = 0.1

        control_data["control_data"]["hoyre_y"] = 10  # Denne variablen styrer ROVens fremdrift. Angir altså hastigheten.
        control_data["control_data"]["manuel"] = 0
        control_data["control_data"]["mROV"] = 0
        control_data["control_data"]["stamp_regulator"] = 1
        control_data["control_data"]["rull_regulator"] = 1
        control_data["control_data"]["depth_regulator"] = 1
        status_data["status_data"].update(control_data["control_data"])
    except Exception as e:
        print("Oops!", e.__class__, "occurred in Auto Init.")
        print(e.args)
        traceback.print_exc()
        return 1
    return 0


