import json, requests

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

r1 = requests.post('http://localhost:5000/post_control_data.json', control_data)
status_data['status_data'].update(r1.json()['status_data'])

r2 = requests.get('http://localhost:5000/get_status_data.json')
status_data['status_data'].update(r2.json()['status_data'])