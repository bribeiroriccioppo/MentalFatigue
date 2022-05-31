"""
Mental Fatigue Questionnaire
"""

#from scipy import signal
#from skimage.morphology import closing
import os
import numpy as np
import pyxdf
import pandas as pd

OFFSET = 1  # seconds

""" def admin_stream2df(stream):
    
    "Convert Admin XDF stream to pandas Dataframe "
    col_names = ['block_start_ts', 'block_end_ts', 'nasa_tlx_start_ts', 'pair_end_time', 'nasa_tlx_end_ts', 'block']
    df = pd.DataFrame(data=stream['time_series'], columns=col_names)
    df['LSL Timestamp'] = stream['time_stamps']
    return df """

def stream2df(stream):
    """ Convert XDF stream to pandas Dataframe """
    col_names = [stream['info']['desc'][0]['channels'][0]['channel'][i]['name'][0]
                 for i in range(stream['time_series'].shape[1])]
    return pd.DataFrame(data=stream['time_series'], columns=col_names)

def load_data(xdf_files):
    """ Load data and basic cleaning"""
    participants = {}

    for XDF in xdf_files:
        #print("Loading data from subject: ", os.path.basename(XDF).split('_')[0])
        data, _ = pyxdf.load_xdf(XDF)

        # Find streams
        for stream in data:
            if stream['info']['name'][0] == 'MentalFatigue_Questionnaire':
                stream_quest = stream

        # Create Questionnaire dataframe
        questionnaire_df = stream2df(stream_quest)

        # Save data
        participants[os.path.splitext(os.path.basename(XDF).split('_')[0])[0]] = {"Questionnaire": questionnaire_df}
    return participants

if (__name__ == "__main__"):
    questionnaire = load_data(['S11_2022-05-30.xdf'])
    print(questionnaire["S11"]['Questionnaire'])
    questionnaire["S11"]['Questionnaire'].to_csv('Questionnaire_Results_S11.csv')

