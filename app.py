import streamlit as st
import time
import pytz
import datetime
import numpy as np
import joblib
from scipy import stats
from PIL import Image
import collections
import pyrebase
import streamlit as st
import pytz
import datetime
import os
import glob
import pandas as pd
import numpy as np
import joblib
from keras.models import model_from_json
from scipy import stats
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from email.message import EmailMessage
import json
import os

#from contextlib import contextmanager, redirect_stdout
#from io import StringIO

def connect_firebase():
    firebaseConfig={  "apiKey": st.secrets["apiKey"],
      "authDomain": st.secrets["authDomain"],
      "databaseURL": st.secrets["databaseURL"],
      "projectId": st.secrets["projectId"],
      "storageBucket": st.secrets["storageBucket"],
      "messagingSenderId": st.secrets["messagingSenderId"],
      "appId": st.secrets["appId"],
      "measurementId": st.secrets["measurementId"] }

    firebase=pyrebase.initialize_app(firebaseConfig)
    
    db = firebase.database()
    #remove events older than 7 days ago
    events = db.child("events").get()
    if events.val() != None:
        for key in dict(events.val()).keys():
            if datetime.datetime.fromtimestamp(int(key)) < datetime.datetime.now()-datetime.timedelta(days=7):
                db.child("events").child(key).remove()
    return db

def load_IPS_models():
    models_path = '/app/nuharips/IPS_models/'
    all_models = glob.glob(os.path.join(models_path , "*.pkl"))
    clfs = []
    for model_path in all_models:
        clf = joblib.load(filename=model_path)
        clfs.append(clf)
    return clfs


def load_HAR_model(model_name):
    if model_name == 'lstm':
        h5_path = '/app/nuharips/HAR_models/lstm51.h5'
        json_path = '/app/nuharips/HAR_models/lstm51.json'
    elif model_name == 'cnn':
        h5_path = '/app/nuharips/HAR_models/cnn51.h5'
        json_path = '/app/nuharips/HAR_models/cnn51.json'
    elif model_name == 'ann':
        h5_path = '/app/nuharips/HAR_models/ann51.h5'
        json_path = '/app/nuharips/HAR_models/ann51.json'

    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_path)
    if model_name=='ann':
        loaded_model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    else:
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(f"Loaded {model_name.upper()} model from disk")

    return loaded_model


def run_HAR():
    readings = db.child("readings").get()
    firebase_df = pd.DataFrame.from_dict(dict(readings.val()), orient='index').reset_index(drop=True)
    firebase_df = firebase_df.drop(columns=['locationTimestamp_since1970', 'locationLatitude', 'locationLongitude', 'locationAltitude', 'locationSpeed', 'locationSpeedAccuracy', 'locationCourse', 'locationCourseAccuracy', 'locationVerticalAccuracy', 'locationHorizontalAccuracy', 'locationFloor', 'accelerometerTimestamp_sinceReboot', 'motionTimestamp_sinceReboot', 'motionAttitudeReferenceFrame', 'motionMagneticFieldX', 'motionMagneticFieldY', 'motionMagneticFieldZ', 'motionHeading', 'motionMagneticFieldCalibrationAccuracy', 'activityTimestamp_sinceReboot', 'activity', 'activityActivityConfidence', 'activityActivityStartDate', 'pedometerStartDate', 'pedometerAverageActivePace', 'pedometerEndDate', 'altimeterTimestamp_sinceReboot', 'altimeterReset', 'altimeterRelativeAltitude', 'altimeterPressure', 'batteryState', 'batteryLevel'])
    firebase_df = firebase_df.drop(columns=['label'], axis=1, errors='ignore')
    firebase_df = firebase_df.dropna()
    firebase_df['timestamp'] =  pd.to_datetime(firebase_df['loggingTime'])
    firebase_df = firebase_df.drop(columns=['loggingTime'])
    cols = firebase_df.columns.difference(['timestamp'])
    firebase_df[cols] = firebase_df[cols].astype(float)
    firebase_df = firebase_df.sort_values(by='timestamp', ascending=True)

    segments = []

    for i in range(0,  firebase_df.shape[0]- n_time_steps, step):  
        accelerometerAccelerationX = firebase_df['accelerometerAccelerationX'].values[i: i + n_time_steps]
        accelerometerAccelerationY = firebase_df['accelerometerAccelerationY'].values[i: i + n_time_steps]
        accelerometerAccelerationZ = firebase_df['accelerometerAccelerationZ'].values[i: i + n_time_steps]
        motionYaw = firebase_df['motionYaw'].values[i: i + n_time_steps]
        motionRoll = firebase_df['motionRoll'].values[i: i + n_time_steps]
        motionPitch = firebase_df['motionPitch'].values[i: i + n_time_steps]
        motionRotationRateX = firebase_df['motionRotationRateX'].values[i: i + n_time_steps]
        motionRotationRateY = firebase_df['motionRotationRateY'].values[i: i + n_time_steps]
        motionRotationRateZ = firebase_df['motionRotationRateZ'].values[i: i + n_time_steps]
        motionUserAccelerationX = firebase_df['motionUserAccelerationX'].values[i: i + n_time_steps]
        motionUserAccelerationY = firebase_df['motionUserAccelerationY'].values[i: i + n_time_steps]
        motionUserAccelerationZ = firebase_df['motionUserAccelerationZ'].values[i: i + n_time_steps]
        motionQuaternionX = firebase_df['motionQuaternionX'].values[i: i + n_time_steps]
        motionQuaternionY = firebase_df['motionQuaternionY'].values[i: i + n_time_steps]
        motionQuaternionZ = firebase_df['motionQuaternionZ'].values[i: i + n_time_steps]
        motionQuaternionW = firebase_df['motionQuaternionW'].values[i: i + n_time_steps]
        motionGravityX = firebase_df['motionGravityX'].values[i: i + n_time_steps]
        motionGravityY = firebase_df['motionGravityY'].values[i: i + n_time_steps]
        motionGravityZ = firebase_df['motionGravityZ'].values[i: i + n_time_steps]
        #pedometerNumberOfSteps = firebase_df['pedometerNumberOfSteps'].values[i: i + n_time_steps]
        #pedometerCurrentPace = firebase_df['pedometerCurrentPace'].values[i: i + n_time_steps]
        #pedometerCurrentCadence = firebase_df['pedometerCurrentCadence'].values[i: i + n_time_steps]
        #pedometerDistance = firebase_df['pedometerDistance'].values[i: i + n_time_steps]
        #pedometerFloorsAscended = firebase_df['pedometerFloorsAscended'].values[i: i + n_time_steps]
        #pedometerFloorsDescended = firebase_df['pedometerFloorsDescended'].values[i: i + n_time_steps]

        #segments.append([accelerometerAccelerationX, accelerometerAccelerationY, accelerometerAccelerationZ, motionYaw, motionRoll, motionPitch, motionRotationRateX, motionRotationRateY, motionRotationRateZ, motionUserAccelerationX, motionUserAccelerationY, motionUserAccelerationZ, motionQuaternionX, motionQuaternionY, motionQuaternionZ, motionQuaternionW, motionGravityX, motionGravityY, motionGravityZ, pedometerNumberOfSteps, pedometerCurrentPace, pedometerCurrentCadence, pedometerDistance, pedometerFloorsAscended, pedometerFloorsDescended])

        segments.append([accelerometerAccelerationX, accelerometerAccelerationY, accelerometerAccelerationZ, motionYaw, motionRoll, motionPitch, motionRotationRateX, motionRotationRateY, motionRotationRateZ, motionUserAccelerationX, motionUserAccelerationY, motionUserAccelerationZ, motionQuaternionX, motionQuaternionY, motionQuaternionZ, motionQuaternionW, motionGravityX, motionGravityY, motionGravityZ])

    
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)
    reshaped_segments = sc_har.transform(reshaped_segments.reshape(-1, reshaped_segments.shape[-1])).reshape(reshaped_segments.shape)
    # loaded_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # loaded_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # loaded_ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    lstm_prediction = loaded_lstm.predict(reshaped_segments)
    cnn_prediction = loaded_cnn.predict(reshaped_segments)
    ann_reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps * n_features)
    ann_prediction = loaded_ann.predict(ann_reshaped_segments)

    lstm_activity = activities[stats.mode(np.argmax(lstm_prediction, axis=1))[0][0]]
    cnn_activity = activities[stats.mode(np.argmax(cnn_prediction, axis=1))[0][0]]
    ann_activity = activities[stats.mode(np.argmax(ann_prediction, axis=1))[0][0]]
    
    return lstm_activity, cnn_activity, ann_activity


def run_IPS():
    esp1_readings = db.child("esp1").get()
    esp2_readings = db.child("esp2").get()
    esp3_readings = db.child("esp3").get()
    temp = [esp1_readings, esp2_readings, esp3_readings]
    frames = []
    for reads in temp:
        if len(reads.val())==1:
            frames.append(reads.val()[0])
        else:
            frames.append(list(next(iter(esp1_readings.val().items())))[1])
    
    dfs = []
    count = 1
    for frame in frames:
        frame = pd.DataFrame.from_dict(dict(frame), orient='index').reset_index(drop=True)
        frame.columns = ['esp' + str(count)]
        dfs.append(frame)
        count+=1
    
    esp_df = pd.concat(dfs, axis=1, ignore_index=False)
    esp_df.dropna(inplace=True)
    esp_df = esp_df[~((esp_df['esp1'] == -110) & (esp_df['esp2'] == -110) & (esp_df['esp3'] == -110))]    
    esp_df = sc_ips.transform(esp_df)

    predictions = []
    for clf in clfs:
        name = type(clf).__name__
        if(name=='SVC'):
            name = name + '_' + clf.get_params()['kernel']
        predictions.append([name, clf.predict(esp_df)[0]])

    predictions_df = pd.DataFrame(predictions, columns=['Classifier','Prediction'])
    predictions_df['Prediction'] = predictions_df['Prediction'].replace(['room_1', 'room_2', 'room_3'], ['Living Room', 'Bedroom', 'Bathroom'])

    return predictions_df


def record_event(ips_pred, har_pred, event):
    event_ts = datetime.datetime.now(pytz.timezone("Africa/Cairo"))
    data = {"Event Timestamp": event_ts.strftime("%d/%m/%Y %H:%M:%S"),
            "Location": ips_pred,
            "Activity": har_pred,
            "Event": event}
    db.child("events").child(int(datetime.datetime.timestamp(event_ts))).set(data)


def get_events():
    try:
        events = db.child("events").get()
        events_df = pd.DataFrame.from_dict(dict(events.val()), orient='index').reset_index(drop=True)
        events_df['Event Timestamp'] =  pd.to_datetime(events_df['Event Timestamp'])
        
        events_df = events_df.sort_values('Event Timestamp', ascending=False)
        events_df['Event Timestamp'] =  events_df['Event Timestamp'].dt.strftime("%d/%m/%Y %H:%M:%S")
        events_df = events_df[['Event Timestamp', 'Location', 'Activity', 'Event']]
        return events_df.head(20)
    except:
        st.warning('No events recorded yet.')

"""
@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield
"""

def gmail_send_message():
    """Create and send an email message
    Print the returned  message id
    Returns: Message object, including message id
    """
    creds = None
    secrets_file = {"token": st.secrets['token'], "refresh_token": st.secrets['refresh_token'], "token_uri": st.secrets['token_uri'], "client_id": st.secrets['client_id'], "client_secret": st.secrets['client_secret'], "scopes": st.secrets['scopes'], "expiry": st.secrets['expiry']}
    secrets_file = json.dumps(secrets_file)
    with open("token.json", "w") as outfile:
        outfile.write(secrets_file)
        
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('gmail', 'v1', credentials=creds)
        message = EmailMessage()

        message.set_content('Alarming activity is detected. Please check the dashboard for more information: https://nuharips.streamlit.app/')

        message['To'] = st.secrets['email']
        message['From'] = 'nuharips@gmail.com'
        message['Subject'] = 'Warning: Alarming Activity Detected'

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()) \
            .decode()

        create_message = {
            'raw': encoded_message
        }
        # pylint: disable=E1101
        send_message = (service.users().messages().send
                        (userId="me", body=create_message).execute())
        print(F'Message Id: {send_message["id"]}')
    except HttpError as error:
        print(F'An error occurred: {error}')
        send_message = None
    return send_message
 
################################################################################################

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']


n_time_steps = 200
n_features = 19
step = 40
n_epochs = 100

activities = ['Downstairs', 'Laying Down', 'Sitting', 'Upstairs', 'Walking']

try:
    db = connect_firebase()
    success1 = st.success("Successfully connected to the database.", icon = "✅")

    clfs = load_IPS_models()
    sc_ips=joblib.load('/app/nuharips/IPS_models/std_scaler.bin')
    success2 = st.success("Successfully loaded IPS models.", icon = "✅")

    loaded_lstm = load_HAR_model('lstm')
    loaded_cnn = load_HAR_model('cnn')
    loaded_ann = load_HAR_model('ann')
    sc_har=joblib.load('/app/nuharips/HAR_models/std_scaler5.bin')
    success3 = st.success("Successfully loaded HAR models.", icon = "✅")
except:
    st.warning('Something went wrong. Please try again later.', icon="⚠️")

# creating a single-element container.
placeholder = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()
refresh_IPS = 'Refreshing...'
refresh_HAR = 'Refreshing...'
prev_har = []
prev_ips = []
EVENTS_RECORDED = 10    #will be 120 for 1 hour
THRESHOLD = 9           #will be 105
SLEEP = 1               #will be 30

while True:
    ips_bool = False
    har_bool = False
    #db.child("esp1").remove()
    #db.child("esp2").remove()
    #db.child("esp3").remove()
    #db.child("readings").remove()
    time.sleep(SLEEP)
    
    with placeholder.container():
        st.write('# Indoor Positioning System')
        st.write('Last Refresh:', refresh_IPS)
        
        try:
            refresh_IPS = datetime.datetime.now(pytz.timezone("Africa/Cairo")).strftime("%d/%m/%Y %H:%M:%S")

            predictions_df = run_IPS()
            ips_pred = np.asarray(predictions_df[predictions_df['Classifier']=='VotingClassifier'])[0][1]
            
            st.write('### Final Prediction:', ips_pred)
            prev_ips.append(ips_pred)
            
            ips_bool = True
            
            if ips_pred == 'Living Room':
                image = Image.open('/app/nuharips/resources/rooms/r1.png')
                st.image(image)
            if ips_pred == 'Bedroom':
                image = Image.open('/app/nuharips/resources/rooms/r2.png')
                st.image(image)
            if ips_pred == 'Bathroom':
                image = Image.open('/app/nuharips/resources/rooms/r3.png')
                st.image(image)
            
        except:
            st.warning('IPS System is Offline!', icon="⚠️")
            

    with placeholder2.container():
        st.write('# Human Activity Recognition')
        st.write('Last Refresh:', refresh_HAR)

        try:
            refresh_HAR = datetime.datetime.now(pytz.timezone("Africa/Cairo")).strftime("%d/%m/%Y %H:%M:%S")
         
            lstm_activity, cnn_activity, ann_activity = run_HAR()
            
            har_pred = stats.mode([lstm_activity, cnn_activity, ann_activity])[0][0]
            st.write("### Final Prediction: ", har_pred)
            prev_har.append(har_pred)
            
            har_bool = True
            
        except:
            st.warning('HAR System is Offline!', icon="⚠️")
    
    placeholder3.empty()
    time.sleep(0.01)
    with placeholder3.container():
        st.write('# Events Record')
        if(ips_bool and har_bool):
            if har_pred == 'Laying Down' and ips_pred == 'Bathroom':
                event = "User is laying down in the Bathroom"
                record_event(ips_pred, har_pred, event)
                st.error('Alarming activity detected!')
                if gmail_send_message()['labelIds'] == ['SENT']:
                    st.error('Supervisor is notified!')                
            
        if len(prev_har) == EVENTS_RECORDED and len(prev_ips) == EVENTS_RECORDED:
            ips_counter = collections.Counter(prev_ips)
            ips_counter = list(ips_counter.most_common(1)[0])
            har_counter = collections.Counter(prev_har)
            har_counter = list(har_counter.most_common(1)[0])
            if ips_counter[1] >= THRESHOLD and har_counter[1] >= THRESHOLD:
                event = "User has been " + har_counter[0] + " in the " + ips_counter[0] + " for " + str(EVENTS_RECORDED*SLEEP/3600) + " hour(s)."
                record_event(ips_pred, har_pred, event)
                st.error('Alarming activity detected!')
                if gmail_send_message()['labelIds'] == ['SENT']:
                    st.error('Supervisor is notified!')  
            prev_har = []
            prev_ips = []
                
        
        


        # Display a static table
        events_df = get_events()
        if events_df is not None:
            hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(events_df)