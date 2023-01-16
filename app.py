from functions import *
import streamlit as st
import time
import pytz
import datetime
import numpy as np
import joblib
from scipy import stats
from PIL import Image
import collections

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
            
            st.write('## Predictions: ')
            st.dataframe(predictions_df)
            st.write('Mode:', predictions_df.mode()['Prediction'][0])
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
            
            st.write('## Predictions: ')
            st.write("LSTM Prediction: ", lstm_activity)
            st.write("CNN Prediction: ", cnn_activity)
            st.write("ANN Prediction: ", ann_activity)
            
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