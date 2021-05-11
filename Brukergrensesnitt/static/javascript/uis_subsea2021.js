var manipulator = 0
var mROV = 0
var start = 0
var man = 1
var initialisert = false
var depth = false
var stamp = false
var roll = false
var manip = false
var lys = false
var mROV_lys = false
var mROV_bryter = false
var manuell = false

var bdg = document.getElementsByClassName("badge")
var chk = document.getElementsByClassName("btn-check")
var brytere = document.getElementsByClassName("btn")


var status_data = {"status_data": {
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
var control_data = {"control_data" : {
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
var control_data_mROV = {"control_data_mROV":{
    "stikke1_x": 12,
    "stikke1_y": 13,
    "stikke2_y": 41,
    "mROV": 0,
    "lys": 0,
    "bryter_ext": 0
}}
var status_data_mROV = {"status_data_mROV":{
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
var regulator = {"regulator" : {
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
var status_regulator = { "status_regulator" : {
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

var timer_1
var set_timeout_time = 0;
var is_close = false;
var teller = 0

var scaling = 50
var scalingPressed = false

const  gamepads = navigator.getGamepads()


   // ----------------------------------------------- //
  // Oppdatering av data mellom Server og Frontside  //
 // ------------------------------------------------//

async function update_status(){
    let data
    let res
    // Startopp prosedyre
    if (!initialisert) {
         res = await fetch("/init_kamera");
         res = await fetch("/get_status_data.json")
        status_data = await res.json();
        initialisert = await true;

    }

    // Kommunikasjon med ROV og mROV.
    if (start == 1) {

        if (man ==0) {
           res = await fetch("/auto");
           status_data = await res.json();
           if (status_data["status_data"]["manuel"]==1){
           man = 1;
           await $("#manuell").click();
           }
           }
        else {
            // Hvis Xbox_1 tilkoblet og manuell
            if (gamepads[0]) {
            var [venstre_x, venstre_y, hoyre_x, hoyre_y, paadrag_venstre, paadrag_hoyre, vinkel_venstre, vinkel_hoyre] = convert(gamepads[0].axes[0].toFixed(2), gamepads[0].axes[1].toFixed(2), gamepads[0].axes[2].toFixed(2), gamepads[0].axes[3].toFixed(2))


            if (gamepads[0].buttons[4].pressed && !scalingPressed) {
                scaling = scaling - 1
                console.log('Decrement scaling')
                scalingPressed = true
                }
            else if (gamepads[0].buttons[5].pressed && !scalingPressed) {
                scaling = scaling + 1;
                console.log('Increment scaling');
                scalingPressed = true;
                }

            if ( !gamepads[0].buttons[4].pressed && !gamepads[0].buttons[5].pressed ) {
                scalingPressed = false;
                }
            if (scaling > 100) {
                scaling = 100
                }
            else if (scaling < 0) {
                scaling = 0
                }

            control_data["control_data"]["skalering"] = scaling.toFixed(2)
            control_data["control_data"]["venstre_x"] = venstre_x.toFixed(2)
            control_data["control_data"]["venstre_y"] = venstre_y.toFixed(2)
            control_data["control_data"]["hoyre_x"] = hoyre_x.toFixed(2)
            control_data["control_data"]["hoyre_y"] = hoyre_y.toFixed(2)

            if (manipulator == 1){
                if (gamepads[1]) {
                     control_data["control_data"]["manip1"] = gamepads[1].axes[0].toFixed(2);
                     control_data["control_data"]["manip2"] = gamepads[1].axes[1].toFixed(2);
                     control_data["control_data"]["manip3"] = gamepads[1].axes[2].toFixed(2);
                     control_data["control_data"]["manip4"] = gamepads[1].axes[3].toFixed(2);
                }
            }

            status_data = await $.post("/post_control_data.json", control_data["control_data"]);
            //status_data = await data.responseJSON;
        }
            else {
                status_data = await $.get("/get_status_data.json");
                //status_data = await data.responseJSON;
            }
        }

        // Oppdatering av variabler til styring/kontroll av programm
        manipulator = status_data["status_data"]["manip_paa"];
        mROV = status_data["status_data"]["mROV"];
        start = status_data["status_data"]["start"];
        man = status_data["status_data"]["manuel"];

         control_data["control_data"]["skalering"] = status_data["status_data"]["skalering"];
         control_data["control_data"]["manuel"] = status_data["status_data"]["manuel"];
         control_data["control_data"]["mROV"] = status_data["status_data"]["mROV"];
         control_data["control_data"]["manip_paa"] = status_data["status_data"]["manipt_paa"];
         control_data["control_data"]["depth_regulator"] = status_data["status_data"]["depth_regulator"];
         control_data["control_data"]["stamp_regulator"] = status_data["status_data"]["stamp_regulator"];
         control_data["control_data"]["rull_regulator"] = status_data["status_data"]["rull_regulator"];
         control_data["control_data"]["lys"] = status_data["status_data"]["lys"];


        if (start==1) {
            if (mROV==1) {
            if (gamepads[1]) {
                control_data_mROV["control_data_mROV"]["stikke1_x"] = gamepads[1].axes[0].toFixed(2);
                control_data_mROV["control_data_mROV"]["stikke1_y"] = gamepads[1].axes[1].toFixed(2);
                control_data_mROV["control_data_mROV"]["stikke2_y"] = gamepads[1].axes[3].toFixed(2);
                control_data_mROV["control_data_mROV"]["mROV"] = mROV;
                status_data_mROV = await $.post("/post_control_data_mROV.json", control_data_mROV["control_data_mROV"]);
                //status_data_mROV = await data.responseJSON;
            }
            else {
                  status_data_mROV = await $.get("/get_status_data_mROV.json");
                  //status_data_mROV = await data.responseJSON;
                  //res = await fetch("/get_status_data.json")
                  //status_data = await res.json();
            }
        }
            // Oppdatering av variabler micro-ROV
            control_data_mROV["control_data_mROV"]["mROV"] = status_data_mROV["status_data_mROV"]["mROV"];
            control_data_mROV["control_data_mROV"]["lys"] = status_data_mROV["status_data_mROV"]["lys"];
            control_data_mROV["control_data_mROV"]["bryter_ext"] = status_data_mROV["status_data_mROV"]["bryter_ext"];




            /////////////////// HOVED ROV STATUSVARIABLER///////////////////////////
            // ROV hovedmotorer  $("#")
            update_circ_bars($("#HVF"), status_data["status_data"]["hvf"])
            update_circ_bars($("#HHF"), status_data["status_data"]["hhf"])
            update_circ_bars($("#HVB"), status_data["status_data"]["hvb"])
            update_circ_bars($("#HHB"), status_data["status_data"]["hhb"])
            update_circ_bars($("#VVF"), status_data["status_data"]["vvf"])
            update_circ_bars($("#VHF"), status_data["status_data"]["vhf"])
            update_circ_bars($("#VVB"), status_data["status_data"]["vvb"])
            update_circ_bars($("#VHB"), status_data["status_data"]["vhb"])

            // Info og status
            bdg.spenn1.innerHTML = status_data["status_data"]["spenn_1"]
            bdg.spenn2.innerHTML = status_data["status_data"]["spenn_2"]

            bdg.strom1.innerHTML = status_data["status_data"]["strom_1"]
            bdg.strom2.innerHTML = status_data["status_data"]["strom_2"]

            bdg.dybd.innerHTML = status_data["status_data"]["dybde"]
            bdg.hoyd.innerHTML = status_data["status_data"]["hoyde"]
            bdg.vanntemp.innerHTML = status_data["status_data"]["temp_vann"]
            bdg.temp.innerHTML = status_data["status_data"]["temp"]

            bdg.tel1.innerHTML = status_data["status_data"]["teller"]
            bdg.tid1.innerHTML = status_data["status_data"]["tid"]

            //bdg.aks.innerHTML = "X: " + status_data["status_data"]["aks_x"] + ", Y: " + status_data["status_data"]["aks_y"] + ", Z: " + status_data["status_data"]["aks_z"] + "."
            bdg.ori.innerHTML = "X: " + status_data["status_data"]["vinkel_x"] + ", Y: " + status_data["status_data"]["vinkel_y"] + ", Z: " + status_data["status_data"]["vinkel_z"] + "."



        /////////////////////// Oppdatering av knappestatus /////////////////////

        // Hvis Stoppsignal fra Server/ROV
        if (start == 0) {
            $("#stop").click()
        }

        // Manipulator
        if (manipulator == 1)
             {manip = true}
        else {manip = false}
        if (manip != document.getElementById("rov_btn_1").checked)
            {document.getElementById("rov_btn_1").checked = manip}
            if (manip) {document.getElementById("mode2").checked = true}
            else {document.getElementById("mode2").checked = false}

        // Manuel/Auto
        if (man == 1){
        manuell = true}
        else {manuell = false}
        if ((manuell !=  document.getElementById("manuell").checked) ||  (manuell ==  document.getElementById("auto").checked))
        { document.getElementById("auto").checked = !manuell
          document.getElementById("manuell").checked = manuell
        }

        // ROV Hovedlys
        if (status_data["status_data"]["lys"] == 1)
             {lys = true}
        else {lys = false}
        if (lys != document.getElementById("rov_btn_2").checked)
            {document.getElementById("rov_btn_2").checked = lys}

        // Dybde regulator oppdater bryter posisjon
        if (status_data["status_data"]["depth_regulator"] == 1)
             {depth = true}
        else {depth = false}
        if (depth != document.getElementById("rov_btn_3").checked)
            {document.getElementById("rov_btn_3").checked = depth}

        // Stamp regulator oppdater bryter posisjon
        if (status_data["status_data"]["stamp_regulator"] == 1)
                      {stamp = true}
        else {stamp = false}
        if (stamp != document.getElementById("rov_btn_4").checked)
            {document.getElementById("rov_btn_4").checked = stamp}

        // Rull regulator oppdater bryter posisjon
        if (status_data["status_data"]["rull_regulator"] == 1)
                     {roll = true}
        else {roll = false}
        if (roll != document.getElementById("rov_btn_5").checked)
            {document.getElementById("rov_btn_5").checked = roll}

        // Av/på lamper til manipulator/microROV
        if (mROV == 1) {
            bdg.mrov_pa.className = "badge badge-success"
            bdg.mrov_av.className = "badge badge-dark"
        }
        else {
            bdg.mrov_pa.className = "badge badge-dark"
            bdg.mrov_av.className = "badge badge-danger"
        }

        // ROV manipulator
        if (manipulator == 1) {
            bdg.man_pa.className = "badge badge-success"
            bdg.man_av.className = "badge badge-dark"
            update_circ_bars($("#M1"), status_data["status_data"]["manip1"])
            update_circ_bars($("#M2"), status_data["status_data"]["manip2"])
            update_circ_bars($("#M3"), status_data["status_data"]["manip3"])
            update_circ_bars($("#M4"), status_data["status_data"]["manip4"])
            }
        else {
            bdg.man_pa.className = "badge badge-dark"
            bdg.man_av.className = "badge badge-danger"
        }


       // Oppdatering av mikro-ROV variabler og knapper
          if (mROV == 1) {
            // Motorer
              update_circ_bars($("#vert_1"), status_data_mROV["status_data_mROV"]["motor_vertikal_1"])
              update_circ_bars($("#vert_2"), status_data_mROV["status_data_mROV"]["motor_vertikal_2"])
              update_circ_bars($("#horis_1"), status_data_mROV["status_data_mROV"]["motor_horisontal_1"])

           // Info og status
              bdg.mstrom1.innerHTML = status_data_mROV["status_data_mROV"]["strom_1"]
              bdg.mstrom2.innerHTML = status_data_mROV["status_data_mROV"]["strom_2"]
              bdg.mtemp.innerHTML = status_data_mROV["status_data_mROV"]["temp"]

            // Sjekk av lysstatus
              if (status_data_mROV["status_data_mROV"]["lys"] == 1 )
                    {mROV_lys = true}
                else {mROV_lys = false}
                if (mROV_lys != document.getElementById("mrov_btn_1").checked)
                    {document.getElementById("mrov_btn_1").checked = mROV_lys}

            // Sjekk av bryter status
              if (status_data_mROV["status_data_mROV"]["bryter_ext"] == 1 )
                    {mROV_bryter = true}
                else {mROV_bryter = false}
                if (mROV_bryter != document.getElementById("mrov_btn_2").checked)
                    {document.getElementById("mrov_btn_2").checked = mROV_bryter;}

            }





        }





    }


    if (!is_close) {
           setTimeout(update_status, set_timeout_time);
    }
    }




   // ----------------------------------------------- //
  // Grafikk til motorer, se også styling.css        //
 // ----------------------------------------------- //


// Omregning prosent til grader.
function percentageToDegrees(percentage) {
    var svar = (percentage / 100) * 360;
    return svar
    }

// Oppdater sirkulær progresjons stolpe
function update_circ_bars(bar,value) {
    // bar = Jquery element
    // to get html element bar[0]
    let left = bar.find('.progress-left .progress-bar');
    let right = bar.find('.progress-right .progress-bar');
    // Oppdater tall inne i sirkel
    bar[0].getElementsByClassName('h7')[0].innerHTML = value
       // Oppdater ring
  if (value <= 0 ){
    var list =  bar[0].getElementsByClassName("progress-bar")
    if (list && list.length > 0 && list[0].className == "progress-bar border-danger") {
        list[0].className = "progress-bar border-primary";
        list[1].className = "progress-bar border-primary";
}
   if (value <= -50) {
       left.css('transform', 'rotate(180deg)');
        right.css('transform', 'rotate(' + percentageToDegrees(value-(-50) )+ 'deg)');}
  else  {
          left.css('transform', 'rotate(' + percentageToDegrees(value) + 'deg)');
          right.css('transform', 'rotate(0deg)');}
}

else {
    var list =  bar[0].getElementsByClassName("progress-bar")
    if (list && list.length > 0 && list[0].className == "progress-bar border-primary") {
        list[0].className = "progress-bar border-danger";
        list[1].className = "progress-bar border-danger";
}
if (value <= 50){
          right.css('transform', 'rotate(' + percentageToDegrees(value) + 'deg)');
            left.css('transform', 'rotate(0deg)');
}
else {            right.css('transform', 'rotate(180deg)');
        left.css('transform', 'rotate(' + percentageToDegrees(value-50) + 'deg)');
}

          };

}






   // ----------------------------------------------- //
  // XBOX Kontrollere                                //
 // ----------------------------------------------- //

// Konverter styringsinput fra et kvadratisk omraade til et sirkulaert omraade
function convert(venstre_x, venstre_y, hoyre_x, hoyre_y) {

    // Finner lengden til pådragsvektorene
    paadrag_venstre = Math.sqrt(Math.pow(venstre_x, 2) + Math.pow(venstre_y, 2))
    paadrag_hoyre = Math.sqrt(Math.pow(hoyre_x, 2) + Math.pow(hoyre_y, 2))

    // Setter maksimal lengde til 1
    if (paadrag_venstre > 1) {
        paadrag_venstre = 1
    }
    if (paadrag_hoyre > 1) {
        paadrag_hoyre = 1
    }

    // Finner vinkelen til pådragsvektorene
    vinkel_venstre = Math.atan2(venstre_y, venstre_x)
    vinkel_hoyre = Math.atan2(hoyre_y, hoyre_x)

    // Beregner nye koordinater fra nye pådragsvektorer
    venstre_x = Math.cos(vinkel_venstre)*(paadrag_venstre/1)
    venstre_y = Math.sin(vinkel_venstre)*(paadrag_venstre/1)
    hoyre_x = Math.cos(vinkel_hoyre)*(paadrag_hoyre/1)
    hoyre_y = Math.sin(vinkel_hoyre)*(paadrag_hoyre/1)

    return [venstre_x, venstre_y, hoyre_x, hoyre_y, paadrag_venstre, paadrag_hoyre, vinkel_venstre, vinkel_hoyre]
}

// Lytter etter Xbox Kontrollere.
window.addEventListener('gamepadconnected', event => {
    console.log('Gamepad connected:')
    console.log(event.gamepad)
})
window.addEventListener('gamepaddisconnected', event => {
    console.log('Gamepad disconnected:')
    console.log(event.gamepad)
})



   // ----------------------------------------------- //
  // Kontroll knapper:                               //
 // ----------------------------------------------- //

//  Start bryter
$("#init").change(function(){
if ($(this).is(':checked')){
timer_1 = setTimeout(update_status, set_timeout_time);
start = 1
is_close = false;
}
else {
    clearInterval(timer_1)
    start = 0
    is_close = true;
}

})

//  Stop bryter
$("#stop").change(function(){
if ($(this).is(':checked')){
    is_close = true;
    start = 0
}
else {
start = 1
is_close = false;

}
})

//  mROV bryter
$("#mode1").change(function(){
if ($(this).is(':checked')){
     mROV = 1
     $.post("/post_control_data.json", {
         "mROV": 1,
         "manip_paa": 0
        },
        function(data, status) {
        status_data = data;
        });
         $.post("/post_control_data_mROV.json", {
         "mROV": 1
        },
        function(data, status) {
        status_data_mROV = data;
        });
        $("#rov_btn_1")[0].checked = false;
}
else {
     mROV = 0
         $.post("/post_control_data.json", {
         "mROV": 0
        },
        function(data, status) {
        status_data = data;
        })
                 $.post("/post_control_data_mROV.json", {
         "mROV": 0
        },
        function(data, status) {
        status_data_mROV = data;
        })
}
})

//  Manuell bryter
$("#manuell").change(function(){
if ($(this).is(':checked')){
     manuell = true
     man = 1
     $.post("/post_control_data.json", {
         "manuel": 1,
        },
        function(data, status) {
        status_data = data;
        });
}
else {
     manuell = false
         $.post("/post_control_data.json", {
         "manuel": 0,
        },
        function(data, status) {
        status_data = data;
        })
}
})

//  Auto bryter
$("#auto").change(function(){
if ($(this).is(':checked')){
     manuell = false
             $.post("/post_control_data.json", {
         "manuel": 0,
        },
        function(data, status) {
        status_data = data;
        });
}
else {
         manuell = true
         $.post("/post_control_data.json", {
         "manuel": 1,
        },
        function(data, status) {
        status_data = data;
        })

}

})

//  Manipulator bryter 1
$("#mode2").change(function(){
if ($(this).is(':checked')){
        $.post("/post_control_data.json", {
         "mROV": 0,
         "manip_paa": 1
        },
        function(data, status) {
        status_data = data;
        });
                 $.post("/post_control_data_mROV.json", {
         "mROV": 0
        },
        function(data, status) {
        status_data_mROV = data;
        });
        $("#rov_btn_1")[0].checked = true

}
else {
     $.post("/post_control_data.json", {
         "manip_paa": 0
        },
        function(data, status) {
        status_data = data;
        });
                $("#rov_btn_1")[0].checked = false;

}
})

//  Manipulator bryter 2
$("#rov_btn_1").change(function() {
    if ($(this).is(':checked')) {
        $.post("/post_control_data.json", {
         "mROV": 0,
         "manip_paa": 1
        },
        function(data, status) {
        status_data = data;
        });
                 $.post("/post_control_data_mROV.json", {
         "mROV": 0
        },
        function(data, status) {
        status_data_mROV = data;
        });
        $("#mode2")[0].checked = true

    }
    else {
    $.post("/post_control_data.json", {
         "manip_paa": 0
        },
        function(data, status) {
        status_data = data;
        });
        $("#mode2")[0].checked = false
    }
})

//  Lys rov
$("#rov_btn_2").change(function(){
if ($(this).is(':checked')){
        $.post("/post_control_data.json", {
         "lys": 1
        },
        function(data, status) {
        console.log(status);
        status_data = data;
        })
}
else {
     $.post("/post_control_data.json", {
         "lys": 0
        },
        function(data, status) {
        status_data = data;
        })
}
})

//  Dybde regulator
$("#rov_btn_3").change(function(){
if ($(this).is(':checked')){
        $.post("/post_control_data.json", {
            "depth_regulator": 1
        },
        function(data, status) {
        status_data = data;
        })
}
else {
     $.post("/post_control_data.json", {
         "depth_regulator": 0
        },
        function(data, status) {
        status_data = data;
        })
}
})

//  Vippe regulator
$("#rov_btn_4").change(function(){
if ($(this).is(':checked')){
        $.post("/post_control_data.json", {
        "stamp_regulator": 1
        },
        function(data, status) {
        status_data = data;
        })
}
else {
     $.post("/post_control_data.json", {
         "stamp_regulator": 0
        },
        function(data, status) {
        status_data = data;
        })
}
})

//  Helnings regulator
$("#rov_btn_5").change(function(){
if ($(this).is(':checked')){
        $.post("/post_control_data.json", {
         "rull_regulator": 1
        },
        function(data, status) {
        status_data = data;
        })
}
else {
     $.post("/post_control_data.json", {
         "rull_regulator": 0
        },
        function(data, status) {
        status_data = data;
        })
}
})

// Lys microROV
$("#mrov_btn_1").change(function(){
if ($(this).is(':checked')){
        $.post("/post_control_data_mROV.json", {
         "lys": 1
        },
        function(data, status) {
        status_data_mROV = data;
        })
}
else {
     $.post("/post_control_data_mROV.json", {
         "lys": 0
        },
        function(data, status) {
        status_data_mROV = data;
        })
}
})

// Strømbryter microROV
$("#mrov_btn_2").change(function(){
if ($(this).is(':checked')){
        $.post("/post_control_data_mROV.json", {
         "bryter_ext": 1
        },
        function(data, status) {
        status_data_mROV = data;
        })
}
else {
     $.post("/post_control_data_mROV.json", {
         "bryter_ext": 0
        },
        function(data, status) {
        status_data_mROV = data;

        })
}
})





   // ----------------------------------------------- //
  // Knapper som kaller bildebehandlingsfunksjoner:  //
 // ----------------------------------------------- //

brytere.namedItem("ta_bilde_frontkamera").onclick = function(){
        $.get("/ta_bilde_frontkamera", function(data, status) {
                        console.log(data)

            var html = data;
            if (status == "success") {
            let myWindow = window.open("", "Bilde", "width=800,height=500");
            myWindow.document.write(data);}
});
}
brytere.namedItem("ta_bilde_havbunnKamera").onclick = function(){
        $.get("/ta_bilde_havbunnKamera", function(data, status) {
                console.log(data)
                            var html = data;
            if (status == "success") {
            let myWindow = window.open("", "Bilde", "width=800,height=500");
            myWindow.document.write(data);}

});
}
brytere.namedItem("ta_bilde_mROV").onclick = function(){
        $.get("/ta_bilde_mROV", function(data, status) {
                    var html = data;
            if (status == "success") {
            let myWindow = window.open("", "Bilde", "width=800,height=500");
            myWindow.document.write(data);}
});
}
brytere.namedItem("slett_siste_bilde").onclick = function(){
        $.get("/slett_siste_bilde", function(data, status) {
                            var html = data;
            if (status == "success") {
            let myWindow = window.open("", "Bilde", "width=800,height=500");
            myWindow.document.write(data);}
});
}
brytere.namedItem("fotomosaikk").onclick = function(){
        $.get("/fotomosaikk", function(data, status) {
        var html = data;
        if (status == "success") {
        let myWindow = window.open("", "Fotomosaikk", "width=750,height=210");
        myWindow.document.write(data);};
});
}
brytere.namedItem("analyser_oversiktsbilde").onclick = function(){
        $.get("/havbunn", function(data, status) {
        var html = data;
        if (status == "success") {
        let myWindow = window.open("", "Havbunn", "width=225,height=450");
        myWindow.document.write(data);};
});
}
brytere.namedItem("analyser_koralrev").onclick = function(){
        $.get("/korall", function(data, status) {
        var html = data;
        if (status == "success") {
        let myWindow = window.open("", "Korallrev", "width=600,height=600");
        myWindow.document.write(data);};
});
}

///////// DEFINISJON AV XBOX KONTROLLERENS KNAPPER  IKKE I BRUK //////////////////////

     /*  const gamepadState = {
            id: gamepads[0].id,
            axes: [
                { Left_x: venstre_x.toFixed(2)},
                { Left_y: venstre_y.toFixed(2)},
                { Angle_left: vinkel_venstre.toFixed(2)},
                { Right_x: hoyre_x.toFixed(2)},
                { Right_y: hoyre_y.toFixed(2)},
                { Angle_right: vinkel_hoyre.toFixed(2)},
            ],
            buttons: [
                { A: gamepads[0].buttons[0].pressed },
                { B: gamepads[0].buttons[1].pressed },
                { X: gamepads[0].buttons[2].pressed },
                { Y: gamepads[0].buttons[3].pressed },
                { LB: gamepads[0].buttons[4].pressed },
                { RB: gamepads[0].buttons[5].pressed },
                { LT: gamepads[0].buttons[6].pressed },
                { RT: gamepads[0].buttons[7].pressed },
                { Back: gamepads[0].buttons[8].pressed },
                { Start: gamepads[0].buttons[9].pressed },
                { LST: gamepads[0].buttons[10].pressed },
                { RST: gamepads[0].buttons[11].pressed },
                { PAD_UP: gamepads[0].buttons[12].pressed },
                { PAD_DOWN: gamepads[0].buttons[13].pressed },
                { PAD_LEFT: gamepads[0].buttons[14].pressed },
                { PAD_RIGHT: gamepads[0].buttons[15].pressed },
            ],
            skalering: [
                {Skalering: scaling},
            ]
        }*/

