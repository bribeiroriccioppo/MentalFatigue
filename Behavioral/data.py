import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy
import pingouin as pg


#replace accidental responses with nan (pressed space, submitting questionnaire before entering any responses)
""" def filter_flukes(subjects, subject):
    
    nasa_tlx_data = subjects[str(subject)]['nasatlx']
    df = nasa_tlx_data.loc[:, (~nasa_tlx_data.columns.isin(['ts', 'block', 'LSL timestamp']))]
    
    for index, row in df.iterrows():
        
        vals = row.to_numpy()
        if all(elem == 10 for elem in vals): #all values are set to 10 (default)
          
            nasa_tlx_data.loc[index,["mental", "physical", "temporal", "performance", "effort", "frustration"]] = np.nan  
        
    return nasa_tlx_data """


def tone_preprocess(subjects, subject):
    
    tone_data = subjects[str(subject)]['tone_events']
    keyboard_data = subjects[str(subject)]['keyboard']
    
    tone_timestamps = tone_data['LSL timestamp'].to_numpy()
    ts_exp_start = tone_timestamps[0]
    ts_exp_end = tone_timestamps[-1]
    
    blocks = {7: "Single1", 8: "Single2", 9: "Dual1", 10: "Dual2", 11: "Multi1", 12: "Multi2"}
    
    tone_data = tone_data.replace({'block': blocks})
    tone_data = tone_data[tone_data['block'].isin(list(blocks.values()))]
    
    #keep only SPACE pressed events
    keyboard_data = keyboard_data[keyboard_data.Event.str.contains("SPACE pressed")] 
    #remove events that take place before first tone or more than 2 seconds after last tone
    keyboard_data = keyboard_data.loc[(keyboard_data['LSL timestamp'] >= ts_exp_start) & (keyboard_data['LSL timestamp'] <= ts_exp_end+2)]
    
    # assign space pressed events within 1.5 seconds following a tone, to that tone
    reaction_window = 1.5 #seconds
    space_ts = np.empty(tone_data['LSL timestamp'].shape)
    space_ts[:] = np.nan
    for idx, tone_ts in enumerate(tone_data['LSL timestamp']):
        for press_ts in keyboard_data['LSL timestamp']:
            if (press_ts>= tone_ts) & (press_ts <= tone_ts+reaction_window):
                space_ts[idx] = press_ts
                
    tone_data['space_ts'] = space_ts
    tone_data['reaction_time'] = tone_data['space_ts'] - tone_data['LSL timestamp']
    
    target_tone_data = tone_data.loc[tone_data['tone_id'] == 0]
    target_tone_data['miss'] = target_tone_data['reaction_time'].isna() #miss: space not pressed after target tone
    standard_tone_data = tone_data.loc[tone_data['tone_id'] == 1]
    standard_tone_data['FP'] = ~standard_tone_data['reaction_time'].isna() #false positives: space pressed on standard tone
    
    subjects[str(subject)]['tone_events'] = tone_data
    subjects[str(subject)]['keyboard'] = keyboard_data
    
    return tone_data, keyboard_data, target_tone_data, standard_tone_data



def preprocess_math(subjects, subject):
    
    math_data = subjects[str(subject)]['math_events']
    math_data.replace(-1, np.nan, inplace=True) #replace all missing data (-1) by nan
    
    blocks = {9: "Dual1", 10: "Dual2", 11: "Multi1", 12: "Multi2"}
    
    math_data = math_data.replace({'block': blocks})
    math_data = math_data[math_data['block'].isin(list(blocks.values()))]
    #drop nans
    math_data = math_data.dropna()
    math_data['solving_time'] = math_data['submit_ts'] - math_data['init_ts']
    subjects[str(subject)]['math_events'] = math_data
    return math_data

