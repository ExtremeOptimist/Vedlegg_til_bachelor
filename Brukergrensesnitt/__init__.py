from flask import Flask, url_for
import logging
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_talisman import Talisman
import os
import json
import cv2
# This file contains configuration, and starts up the required applications.
# It is the centerpiece of the program. Some code has been commented out due to development mode.


# Configuration of Flask_
app = Flask(__name__)
# Logging to log file
'''logging.basicConfig(filename='logging.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')'''

# Configuration of database via SQLAlchemy
    # From JSON file, Used on deployed server:
#with open('/etc/config.json') as config_file:
#    config = json.load(config_file)

app.config["SECRET_KEY"] = "Hemmlig_kode_her_ikke_les_den_er_hemmelig" #os.environ.get('SECRET_KEY')  # Using environ variable to hide information config.get('SECRET_KEY') #
app.config['SQLALCHEMY_DATABASE_URI'] =  'sqlite:///brukere.sqlite3'                 ##config.get('SQLALCHEMY_DATABASE_URI') #   'mysql+pymysql://jenstrydal:passord@localhost/mydatabase'    os.environ.get('SQLALCHEMY_DATABASE_URI')
app.config["SQLALCHEMY_TRACK_MODIFICATION"] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0



# Talisman sikkerhetskonfigurasjon er ikke implementert på utviklings server
'''
csp = {
'default-src': ['*'],
'script.js-src':  ['*'],
'img-src':  ['*'],
'base-uri':  ['*'],
'form-action':  ['*'],
'connect-src':  ['*'],
'style-src':  ['*'],
'frame-src':  ['*'],
'font-src':  ['*']
}

'''
'''
csp = {'default-src': ['*',


                    "https://fonts.googleapis.com/css?family=Raleway",
                    'https://code.jquery.com/',
                    'https://ajax.googleapis.com/',
                    'https://stackpath.bootstrapcdn.com',
                    'https://cdn.jsdelivr.net/',
                    'https://fonts.googleapis.com',
                    'https://stackpath.bootstrapcdn.com',
                    'https://cdn.jsdelivr.net'],
    'script.js-src': ['\'self\'',
                      'https://code.jquery.com/',
                      'https://ajax.googleapis.com/',
                      'https://stackpath.bootstrapcdn.com',
                      'https://cdn.jsdelivr.net/'],
    'connect-src': ['\'self\''],
    'img-src': ['\'self\''],
    'style-src': ['\'self\'',
                  'https://fonts.googleapis.com',
                  'https://stackpath.bootstrapcdn.com',
                  'https://cdn.jsdelivr.net'],
    'frame-src': ['\'self\'', 'https://fonts.gstatic.com'],
    'font-src': ['\'self\'', 'https://www.google.com'],
    'report-uri': 'https://601ed82b74246c4998ba101d.endpoint.csper.io/',
    'base-uri': '\'self\'',
    'form-action': '\'self\''}


talisman = Talisman(app, content_security_policy=csp)
talisman.content_security_policy_report_only=True
content_security_policy_report_uri='https://601694c151f813be0fcd0652.endpoint.csper.io'



#### THIS HAS ALLREADY BEEN DONE BY TALISMAN #####
app.config['SESSION_COOKIE_SECURE'] = True # oppdater denne når deplot # Secure limits cookies to HTTPS traffic only.
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Strict'
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=5) # NOT set, we only use remember med cookie
# app.config['SESSION_REFRESH_EACH_REQUEST'] = True

# Remember me configurations
app.config['REMEMBER_COOKIE_SECURE'] = True ## oppdater denne når deplot
app.config['REMEMBER_COOKIE_HTTPONLY'] = True

# app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30) # OBS - Creates cookie
# app.config['REMEMBER_COOKIE_REFRESH_EACH_REQUEST'] = True         - Creats cookie
####################################
#talisman.force_https = False ## Kommenter ut når deploy
#talisman.force_https_permanent = False# Kommenter ut når deploy

'''
'''
'''
#  SQLAlchemy database object
db = SQLAlchemy(app)

# Starting Bcrypt password hashing
bcrypt = Bcrypt()

# Login manager
login_manager = LoginManager(app)  # Initialising login manager
login_manager.login_view = "login"  # Showing where login is done so it can redirect to login
login_manager.login_message_category = "info"  # Making nice flash messages
login_manager.refresh_view = 'refresh'
login_manager.needs_refresh_message = 'You need to login again! '
login_manager.needs_refresh_message_category = "info"
login_manager.session_protection = "basic"  # IP and user agent is stored in an hashed identifier.
                                            # Strong -Entire session is deleted if identifier does not match
                                            # Basic - Session set to not fresh


# Importing from other files so that main applications know about them.
# Doing the import here to avoid circular importing.
import Brukergrensesnitt.routes
