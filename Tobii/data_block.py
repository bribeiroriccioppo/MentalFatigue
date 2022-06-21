# Copyright (c) 2021
# Manuel Cherep <mcherep@logitech.com>

"""
Pupil pre-processing of the data.
"""

from scipy import signal
from skimage.morphology import closing
import os
import numpy as np
import pyxdf
import pandas as pd

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


def find_blinks(pupils):
    # TODO: Move to external module
    """ Find the indices of blinks and outliers in the data """
    pupil_gradients = np.abs(np.gradient(pupils))
    pupil_closing = closing(pupil_gradients, np.ones(5))
    pupil_closing[pupil_closing < 0.02] = 0
    blinks_idx = np.where(pupil_closing > 0)[0]
    return blinks_idx


def load_data(xdf_files):
    """ Load data and basic cleaning"""
    participants = {}

    for XDF in xdf_files:
        data, _ = pyxdf.load_xdf(XDF)

        # Find streams
        for stream in data:
            if stream['info']['name'][0] == 'Tobii':
                stream_tobii = stream
            elif stream['info']['name'][0] == 'MentalFatigue_Blocks':
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

        # Create tone task dataframe
        tone_tasks_df = stream2df(stream_mentalfatigue_tone_task)
        keyboard_df = streamkeyboard2df(stream_keyboard)
        keyboard_df = keyboard_df[keyboard_df.events == 'SPACE pressed']

        # Create blocks dataframe
        blocks_df = stream2df(stream_mentalfatigue_blocks)
        #print(blocks_df)
        block_duration = blocks_df['end_ts'] - blocks_df['init_ts']
        blocks_df['lsl_end_ts'] = stream_mentalfatigue_blocks['time_stamps']
        blocks_df['lsl_init_ts'] = blocks_df['lsl_end_ts'] - block_duration
        blocks_df.drop(columns=['init_ts', 'end_ts'], inplace=True)

        # Create questionnaire dataframe
        quest_df = stream2df(stream_mentalfatigue_quest)

        # Create Tobii dataframe
        tobii_df = stream2df(stream_tobii)
        tobii_df['lsl_ts'] = stream_tobii['time_stamps']
        tobii_df = tobii_df[['right_pupil', 'left_pupil',
                             'lsl_ts']].dropna().reset_index(drop=True)

        # Remove blinks
        left_blinks = find_blinks(tobii_df.left_pupil.values)
        right_blinks = find_blinks(tobii_df.right_pupil.values)
        blinks = set(left_blinks.tolist() + right_blinks.tolist())
        tobii_df.drop(blinks, inplace=True)

        # Low pass filter
        sos = signal.butter(1, 4, 'lp', fs=90, output='sos')
        left_pupil_lp = signal.sosfiltfilt(sos, tobii_df.left_pupil.values)
        tobii_df['left_pupil'] = left_pupil_lp
        right_pupil_lp = signal.sosfiltfilt(sos, tobii_df.right_pupil.values)
        tobii_df['right_pupil'] = right_pupil_lp

        # Include the block, task and condition in the pupil data
        tobii_df['block_idx'] = np.nan
        tobii_df['block_id'] = np.nan
        tobii_df['task_idx'] = np.nan

        for _, row in blocks_df.iterrows():
            # Include block index and id in Tobii
            tobii_row_indices = ((tobii_df['lsl_ts'] >= row['lsl_init_ts']) & (
                tobii_df['lsl_ts'] <= row['lsl_end_ts']))
            tobii_df.loc[tobii_row_indices, 'block_idx'] = row['block_idx']
            tobii_df.loc[tobii_row_indices, 'block_id'] = row['block_id']

        for _, row in math_tasks_df.iterrows():
            # Include task index and condition in Tobii
            tobii_row_indices = ((tobii_df['lsl_ts'] >= row['lsl_init_ts']) & (
                tobii_df['lsl_ts'] <= row['lsl_end_ts']))
            tobii_df.loc[tobii_row_indices, 'task_idx'] = row['task_idx']
        
        """ for _, row in tone_tasks_df.iterrows():
            # Include task index and condition in Tobii
            tobii_row_indices = ((tobii_df['lsl_ts'] >= row['lsl_init_ts']) & (
                tobii_df['lsl_ts'] <= row['lsl_end_ts']))
            tobii_df.loc[tobii_row_indices, 'task_idx'] = row['task_idx'] """
        
        # Delete irrelevant data
        tobii_df.drop(tobii_df[tobii_df.block_idx.isna()].index,
                      inplace=True)

        # Include pupil dilation percentage
        tobii_df['left_pupil_pct'] = 0
        tobii_df['right_pupil_pct'] = 0

        #baseline = tobii_df.loc[tobii_df['block_idx'] == 1.0].median()
        
        for _, row in blocks_df.iterrows():
            tobii_row_indices = ((tobii_df['lsl_ts'] >= row['lsl_init_ts']) & (tobii_df['lsl_ts'] <= row['lsl_end_ts']))
            block_tobii_df = tobii_df[tobii_row_indices]
            baseline = block_tobii_df.head(9).median()
            tobii_df.loc[tobii_row_indices,
                         'left_pupil_pct'] = block_tobii_df['left_pupil'] / baseline.left_pupil
            tobii_df.loc[tobii_row_indices,
                         'right_pupil_pct'] = block_tobii_df['right_pupil'] / baseline.right_pupil

        tobii_df['pupil_pct'] = tobii_df.loc[:,
                                             "left_pupil_pct":"right_pupil_pct"].median(axis=1)
        tobii_df['pupil'] = tobii_df.loc[:,
                                         "right_pupil":"left_pupil"].median(axis=1)

        # Save data
        participants[os.path.splitext(os.path.basename(XDF))[0]] = {"Tobii": tobii_df,
                                                                    "MathTasks": math_tasks_df,
                                                                    "ToneTasks": tone_tasks_df,
                                                                    "Keyboard": keyboard_df,
                                                                    "Blocks": blocks_df,
                                                                    "Questionnaire": quest_df}

    return participants


def preprocessing(xdf_files):
    participants = load_data(xdf_files)
    return participants