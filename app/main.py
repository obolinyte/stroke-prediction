from flask import Flask, render_template, jsonify, request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, BooleanField, SubmitField, FloatField
from wtforms.validators import ValidationError, DataRequired, NumberRange, Email, EqualTo, Length, Optional
from custom_transformers import *
import joblib

app = Flask(__name__, template_folder='.')
app.config['SECRET_KEY'] = 'LongAndRandomSecretKey'
stroke_model = joblib.load("models/stroke_classifier.pkl")
hyper_model = joblib.load("models/hyper_classifier.pkl")
glucose_model = joblib.load("models/glucose_regressor.pkl")
bmi_model = joblib.load("models/bmi_regressor.pkl")


class FormBase(FlaskForm):
    gender = SelectField(label=('Gender'), choices=['male', 'female'], validators=[DataRequired()], )
    age = FloatField(label=('Age'), validators=[DataRequired(message='Invalid "Age" input. Only numbers allowed'), NumberRange(min=0, max=83)],render_kw={"placeholder": "enter age"})
    heart_disease = SelectField(label=('Heart disease'), choices=['no', 'yes'], validators=[DataRequired()])
    ever_married = SelectField(label=('Ever married'), choices=['no', 'yes'], validators=[DataRequired()])
    work_type = SelectField(label=('Work type'), choices=['private', 'self-employed', 'govt_job', 'never_worked'],
                            validators=[DataRequired()])
    smoking_status = SelectField(label=('Smoking Status'),
                                 choices=['never smoked', 'formerly smoked', 'smokes', 'unknown'],
                                 validators=[DataRequired()])
    residence_type = SelectField(label=('Residence type'), choices=['rural', 'urban'], validators=[DataRequired()])
    submit = SubmitField(label=('Submit'))


class CreateStrokeForm(FormBase):
    hypertension = SelectField(label=('Hypertension'), choices=['no', 'yes'], validators=[DataRequired()])
    avg_glucose_level = FloatField(label=('Average glucose level'), validators=[DataRequired(message='Invalid "Average glucose level" input. Only numbers allowed.'), NumberRange(min=55, max=272)])
    bmi = FloatField(label=('Body Mass Index (Optional)'), validators=[Optional(), DataRequired(message='Invalid "Body Mass Index" input. Only numbers allowed.'), NumberRange(min=10, max=98)])


class CreateHypertensionForm(FormBase):
    avg_glucose_level = FloatField(label=('Average glucose level'),
                                   validators=[DataRequired(message='Invalid "Average glucose level" input. Only numbers allowed.'), NumberRange(min=55, max=272)])
    bmi = FloatField(label=('Body Mass Index (Optional)'), validators=[Optional(), DataRequired(message='Invalid "Body Mass Index" input. Only numbers allowed.'), NumberRange(min=10, max=98)])


class CreateGlucoseForm(FormBase):
    hypertension = SelectField(label=('Hypertension'), choices=['no', 'yes'], validators=[DataRequired()])
    bmi = FloatField(label=('Body Mass Index (Optional)'), validators=[Optional(), DataRequired(message='Invalid "Body Mass Index" input. Only numbers allowed'), NumberRange(min=10, max=98)])


class CreateBmiForm(FormBase):
    hypertension = SelectField(label=('Hypertension'), choices=['no', 'yes'], validators=[DataRequired()])
    avg_glucose_level = FloatField(label=('Average glucose level'),
                                   validators=[DataRequired(message='Invalid "Average glucose level" input. Only numbers allowed.'), NumberRange(min=55, max=272)])


@app.route('/', methods=('GET', 'POST'))
def render_home():
    return render_template('templates/index.html',)


@app.route('/stroke-form', methods=('GET', 'POST'))
def render_stroke_form():
    form = CreateStrokeForm(csrf_enabled=True)
    if form.validate_on_submit():
        features = request.form.to_dict()
        if features['bmi'] == '':
            features['bmi'] = np.nan
        del features['csrf_token']
        del features['submit']
        final_features = pd.DataFrame(features, index=[0])

        for probs, pred in zip(stroke_model.predict_proba(final_features), stroke_model.predict(final_features)):
            prediction_text = "You have a risk of stroke."
            if not pred:
                prediction_text = "You don't have a risk of stroke!"
            res = {'prediction': {'text': prediction_text}, 'probability': {'value': round(probs[pred], 2),
                                                                            'text': 'Probability:'}}
            return render_template('templates/form.html', form=form, result=res, message='Stroke')
    return render_template('templates/form.html', form=form, message='Stroke')


@app.route('/hypertension-form', methods=('GET', 'POST'))
def render_hypertension_form():
    form = CreateHypertensionForm(csrf_enabled=True)
    if form.validate_on_submit():
        features = request.form.to_dict()
        if features['bmi'] == '':
            features['bmi'] = np.nan
        del features['csrf_token']
        del features['submit']
        final_features = pd.DataFrame(features, index=[0])

        for probs, pred in zip(hyper_model.predict_proba(final_features), hyper_model.predict(final_features)):
            prediction_text = "You have a risk of hypertension."
            if not pred:
                prediction_text = "You don't have a risk of hypertension!"
            res = {'prediction': {'text': prediction_text}, 'probability': {'value': round(probs[pred], 2),
                                                                            'text': 'Probability:'}}
            return render_template('templates/form.html', form=form, result=res, message='Hypertension')
    return render_template('templates/form.html', form=form, message='Hypertension')


@app.route('/glucose-form', methods=('GET', 'POST'))
def render_glucose_form():
    form = CreateGlucoseForm(csrf_enabled=True)
    if form.validate_on_submit():
        features = request.form.to_dict()
        if features['bmi'] == '':
            features['bmi'] = np.nan
        del features['csrf_token']
        del features['submit']
        final_features = pd.DataFrame(features, index=[0])
        pred = glucose_model.predict(final_features)
        prediction_text = 'Estimated avg glucose level: '
        res = {'prediction': {'value': round(pred[0], 2), 'text': prediction_text}}
        return render_template('templates/form.html', form=form, result=res, message='Glucose level')
    return render_template('templates/form.html', form=form, message='Glucose level')


@app.route('/bmi-form', methods=('GET', 'POST'))
def render_bmi_form():
    form = CreateBmiForm(csrf_enabled=True)
    if form.validate_on_submit():
        features = request.form.to_dict()
        del features['csrf_token']
        del features['submit']
        final_features = pd.DataFrame(features, index=[0])
        pred = bmi_model.predict(final_features)
        prediction_text = 'Estimated BMI: '
        res = {'prediction': {'value': round(pred[0], 2), 'text': prediction_text}}
        return render_template('templates/form.html', form=form, result=res, message='Body Mass Index')
    return render_template('templates/form.html', form=form, message='Body Mass Index')


@app.route('/hyper-glucose-form', methods=('GET', 'POST'))
def render_hyper_glucose_form():
    form = CreateStrokeForm(csrf_enabled=True)
    if form.validate_on_submit():
        features = request.form.to_dict()
        if features['bmi'] == '':
            features['bmi'] = np.nan
        del features['csrf_token']
        del features['submit']

        final_features = pd.DataFrame(features, index=[0])
        for probs, pred in zip(hyper_model.predict_proba(final_features.drop(columns='hypertension')),
                               hyper_model.predict(final_features.drop(columns='hypertension'))):
            hyper_pred_text = "You have a risk of hypertension."
            if not pred:
                hyper_pred_text = "You don't have a risk of hypertension!"

        glucose_pred = glucose_model.predict(final_features.drop(columns='avg_glucose_level'))
        glucose_text = 'Estimated avg glucose level: '

        res = {'hyper_prediction': {'hyper_text': hyper_pred_text, 'hyper_prob': round(probs[pred], 2),
                                    'hyper_prob_text': 'Probability: '},
               'glucose_prediction': {'glucose_text': glucose_text, 'glucose_pred': round(glucose_pred[0], 2)}}

        return render_template('templates/multioutput_form.html', form=form, result=res,
                               message='Hypertension & glucose level')
    return render_template('templates/multioutput_form.html', form=form, message='Hypertension & glucose level')


@app.route('/hyper-bmi-form', methods=('GET', 'POST'))
def render_hyper_bmi_form():
    form = CreateStrokeForm(csrf_enabled=True)
    if form.validate_on_submit():
        features = request.form.to_dict()
        if features['bmi'] == '':
            features['bmi'] = np.nan
        del features['csrf_token']
        del features['submit']

        final_features = pd.DataFrame(features, index=[0])

        for probs, pred in zip(hyper_model.predict_proba(final_features.drop(columns='hypertension')),
                               hyper_model.predict(final_features.drop(columns='hypertension'))):
            hyper_pred_text = "You have a risk of hypertension."
            if not pred:
                hyper_pred_text = "You don't have a risk of hypertension!"

        bmi_pred = bmi_model.predict(final_features.drop(columns='bmi'))
        bmi_text = 'Estimated BMI: '

        res = {'hyper_prediction': {'hyper_text': hyper_pred_text, 'hyper_prob': round(probs[pred], 2), 'hyper_prob_text':'Probability: '},
               'bmi_prediction': {'bmi_text': bmi_text, 'bmi_pred': round(bmi_pred[0], 2)}}

        return render_template('templates/multioutput_form.html', form=form, result=res, message='Hypertension & BMI')
    return render_template('templates/multioutput_form.html', form=form, message='Hypertension & BMI')


@app.route('/glucose-bmi-form', methods=('GET', 'POST'))
def render_glucose_bmi_form():
    form = CreateStrokeForm(csrf_enabled=True)
    if form.validate_on_submit():
        features = request.form.to_dict()
        if features['bmi'] == '':
            features['bmi'] = np.nan
        del features['csrf_token']
        del features['submit']

        final_features = pd.DataFrame(features, index=[0])

        glucose_pred = glucose_model.predict(final_features.drop(columns='avg_glucose_level'))
        glucose_text = 'Estimated avg glucose level: '

        bmi_pred = bmi_model.predict(final_features.drop(columns='bmi'))
        bmi_text = 'Estimated BMI: '

        res = {'glucose_prediction': {'glucose_text': glucose_text, 'glucose_pred': round(glucose_pred[0], 2)},
               'bmi_prediction': {'bmi_text': bmi_text, 'bmi_pred': round(bmi_pred[0], 2)}}

        return render_template('templates/multioutput_form.html', form=form, result=res, message='Glucose level & BMI')
    return render_template('templates/multioutput_form.html', form=form, message='Glucose level & BMI')


@app.route('/hyper_glucose-bmi-form', methods=('GET', 'POST'))
def render_hyper_glucose_bmi_form():
    form = CreateStrokeForm(csrf_enabled=True)
    if form.validate_on_submit():
        features = request.form.to_dict()
        if features['bmi'] == '':
            features['bmi'] = np.nan
        del features['csrf_token']
        del features['submit']

        final_features = pd.DataFrame(features, index=[0])

        for probs, pred in zip(hyper_model.predict_proba(final_features.drop(columns='hypertension')),
                               hyper_model.predict(final_features.drop(columns='hypertension'))):
            hyper_pred_text = "You have a risk of hypertension."
            if not pred:
                hyper_pred_text = "You don't have a risk of hypertension!"

        glucose_pred = glucose_model.predict(final_features.drop(columns='avg_glucose_level'))
        glucose_text = 'Estimated avg glucose level: '

        bmi_pred = bmi_model.predict(final_features.drop(columns='bmi'))
        bmi_text = 'Estimated BMI: '

        res = {'hyper_prediction': {'hyper_text': hyper_pred_text, 'hyper_prob': round(probs[pred], 2), 'hyper_prob_text':'Probability: '},
               'glucose_prediction': {'glucose_text': glucose_text, 'glucose_pred': round(glucose_pred[0], 2)},
               'bmi_prediction': {'bmi_text': bmi_text, 'bmi_pred': round(bmi_pred[0], 2)}}

        return render_template('templates/multioutput_form.html', form=form, result=res, multi_message='Hypertension, glucose level & BMI')
    return render_template('templates/multioutput_form.html', form=form, multi_message='Hypertension, glucose level & BMI')


if __name__ == '__main__':
    app.run(debug=True)