from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField, SelectField, HiddenField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError, NumberRange


class LoginForm(FlaskForm):
    name = StringField('Brukernavn', validators=[DataRequired(), Length(min=2, max=50)])
    password = PasswordField('Passord', validators=[DataRequired(), Length(min=8, max=100)])
    remember = BooleanField('Husk Meg')
    submit = SubmitField('Logg inn')


