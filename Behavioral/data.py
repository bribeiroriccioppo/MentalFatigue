from scipy import signal
from skimage.morphology import closing
import os
import numpy as np
import pyxdf
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pingouin as pg

def flatten(l):
    return [item for sublist in l for item in sublist]

def streamkeyboard2df(stream):
    return pd.DataFrame({'events': flatten(stream['time_series']), 'reaction_ts': stream['time_stamps']})

def stream2df(stream):
    # TODO: Move to external module
    """ Converte XDF stream to pandas Dataframe """
    col_names = [stream['info']['desc'][0]['channels'][0]['channel'][i]['name'][0]
                 for i in range(stream['time_series'].shape[1])]
    return pd.DataFrame(data=stream['time_series'], columns=col_names)

def load_data(xdf_files):
    """ Load data and basic cleaning"""
    participants = {}

    for XDF in xdf_files:
        data, _ = pyxdf.load_xdf(XDF)

        # Find streams
        for stream in data:
            if stream['info']['name'][0] == 'MentalFatigue_Blocks':
                #print(stream['time_series'])
                stream_mentalfatigue_blocks = stream
            elif stream['info']['name'][0] == 'MentalFatigue_Math_Task':
                stream_mentalfatigue_math_task = stream
            elif stream['info']['name'][0] == 'MentalFatigue_Tone_Task':
                stream_mentalfatigue_tone_task = stream
            elif stream['info']['name'][0] == 'MentalFatigue_Questionnaire':
                stream_mentalfatigue_quest = stream
            elif stream['info']['name'][0] == 'Keyboard':
                stream_keyboard = stream
        
        # Create math task dataframe
        math_tasks_df = stream2df(stream_mentalfatigue_math_task)
        math_task_duration = math_tasks_df['end_ts'] - math_tasks_df['init_ts']
        math_tasks_df['lsl_end_ts'] = stream_mentalfatigue_math_task['time_stamps']
        math_tasks_df['lsl_init_ts'] = math_tasks_df['lsl_end_ts'] - math_task_duration
        math_tasks_df.drop(columns=['init_ts', 'end_ts'], inplace=True)
        # Create tone task + keyboard dataframe
        tone_tasks_df = stream2df(stream_mentalfatigue_tone_task)
        tone_tasks_df['lsl_stimulus_ts'] = stream_mentalfatigue_tone_task['time_stamps']
        #print(tone_tasks_df)
        keyboard_df = streamkeyboard2df(stream_keyboard)
        keyboard_df = keyboard_df[keyboard_df.events == 'SPACE pressed']

        reaction_window = 1.5 #seconds
        space_ts = np.empty(tone_tasks_df['stimulus_ts'].shape)
        space_ts[:] = np.nan
        for idx, tone_ts in enumerate(tone_tasks_df['stimulus_ts']):
            for press_ts in keyboard_df['reaction_ts']:
                if (press_ts>= tone_ts) & (press_ts <= tone_ts+reaction_window):
                    space_ts[idx] = press_ts
        
        tone_tasks_df['space_ts'] = space_ts
        tone_tasks_df['rt'] = tone_tasks_df['space_ts'] - tone_tasks_df['stimulus_ts']

        """ 
        def tone_preprocess(subject):
            tone_data = subjects[str(subject)]['tone_events']
            keyboard_data = subjects[str(subject)]['keyboard']
            
            tone_timestamps = tone_data['LSL timestamp'].to_numpy()
            ts_exp_start = tone_timestamps[0]
            ts_exp_end = tone_timestamps[-1]
            
            #blocks = {2: "Practice", 7: "Single1", 8: "Single2", 9: "Dual1", 10: "Dual2", 11: "Multi1", 12: "Multi2"}
            blocks = {7: "Single1", 8: "Single2", 9: "Dual1", 10: "Dual2", 11: "Multi1", 12: "Multi2"}
            tone_data = tone_data.replace({'block': blocks})
            #print(list(blocks.values()))
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
            
            return tone_data, keyboard_data, target_tone_data, standard_tone_data """

        # Create blocks dataframe
        blocks_df = stream2df(stream_mentalfatigue_blocks)
        #print(blocks_df)
        block_duration = blocks_df['end_ts'] - blocks_df['init_ts']
        blocks_df['lsl_end_ts'] = stream_mentalfatigue_blocks['time_stamps']
        blocks_df['lsl_init_ts'] = blocks_df['lsl_end_ts'] - block_duration
        blocks_df.drop(columns=['init_ts', 'end_ts'], inplace=True)

        # Create questionnaire dataframe
        quest_df = stream2df(stream_mentalfatigue_quest)

        # Save data
        participants[os.path.splitext(os.path.basename(XDF))[0]] = {"MathTasks": math_tasks_df,
                                                                    "ToneTasks": tone_tasks_df,
                                                                    "Keyboard": keyboard_df,
                                                                    "Blocks": blocks_df,
                                                                    "Questionnaire": quest_df}

    return participants

def preprocessing(xdf_files):
    participants = load_data(xdf_files)
    return participants

def tone_preprocess(subjects, subject):
    
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