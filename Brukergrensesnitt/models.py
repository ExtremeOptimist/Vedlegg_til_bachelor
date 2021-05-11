from Brukergrensesnitt import db
from flask_login import UserMixin

# This file contains classes and functions concerning the login database.
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    name = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return self.name



def kodelinjer_til_opprettelse_av_db():
    from Brukergrensesnitt import db,bcrypt
    from Brukergrensesnitt.models import User

    db.create_all()
    db.session.add(User(id=1, name="admin", password=bcrypt.generate_password_hash("passord").decode("utf-8")))
    db.session.commit()

def krypter_pass():
    # åpner jsonfil med passord i klartekst. Hash'er passordene.
    import json
    from flask_bcrypt import Bcrypt
    bcrypt = Bcrypt()
    with open("Github/GUI/Flask_1/login.json") as read_file:
        brukere = json.load(read_file)
    for i in range(len(brukere["Brukere"])):
        brukere["Brukere"][f'{i+1}']["passord"] = bcrypt.generate_password_hash(brukere["Brukere"][f'{i+1}']["passord"]).decode("utf-8")
    with open("Github/GUI/Flask_1/login.json","w") as write_file:
        json.dump(brukere,write_file)

