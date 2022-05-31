# Copyright (c) 2022
# Manuel Cherep <mcherep@logitech.com>

"""
EEG pre-processing of the data.
"""

from mne.time_frequency import psd_multitaper
import mne
import numpy as np
import os
import pyxdf
import pandas as pd

NAMES_1020 = ['C4', 'C2', 'Cz', 'C1', 'C3', 'FT8', 'TP8', 'FT7', 'TP7']
BAD_CHANNELS = ['C2', 'Cz', 'C1', 'C3', 'FT8', 'FT7', 'TP7',
                'F8', 'F3', 'F4', 'P4', 'P3', 'T3', 'T4']  # Unused channels
""" BAD_CHANNELS = ['C4', 'C2', 'Cz', 'C1', 'C3', 'FT8', 'FT7',
                'F8', 'F3', 'F4', 'P4', 'P3', 'T3', 'T4']  # Unused channels """

CHANNELS = {'All': ['C4', 'TP8'],
            'C4': ['C4'],
            'TP8': ['TP8']}
""" CHANNELS = {'All': ['TP7', 'TP8'],
            'TP7': ['TP7'],
            'TP8': ['TP8']} """


FREQ_BANDS = [('Delta', 1, 3),
              ('Theta', 4, 7),
              ('Alpha1', 8, 10),
              ('Alpha2', 11, 13),
              ('Beta', 14, 30),
              ('Gamma', 31, 45)]
EVENTS = {'0.0': 0, '1.0': 1, '2.0': 2, '3.0': 3, '6.0': 6, '8.0': 8, '10.0': 10,
          '12.0': 12, '13.0': 13, '14.0': 14, '15.0': 15, '16.0': 16, '17.0': 17, 
          '18.0': 18, '19.0': 19, '20.0': 20, '21.0': 21, '22.0': 22, '23.0': 23, 
          '24.0': 24, '25.0': 25, '26.0': 26, '27.0': 27, '28.0': 28, '29.0': 29,
          '30.0': 30, '31.0': 31, '32.0': 32, '33.0': 33, '34.0': 34, '35.0': 35,
          '36.0': 36, '37.0': 37, '38.0': 38, '39.0': 39, '40.0': 40}


def stream2df(stream):
    """ Convert XDF stream to pandas Dataframe """
    col_names = [stream['info']['desc'][0]['channels'][0]['channel'][i]['name'][0]
                 for i in range(stream['time_series'].shape[1])]
    return pd.DataFrame(data=stream['time_series'], columns=col_names)


def stream2mne(stream):
    """
    Adapted from https://github.com/cbrnr/mnelab/blob/main/mnelab/io/xdf.py
    We cannot use it because we don't have a XDF file at this point.
    """

    n_chans = int(stream["info"]["channel_count"][0])
    fs = float(stream["info"]["nominal_srate"][0])
    origin_time = stream['time_stamps'][0]

    # Search for labels and units in the stream;
    # create otherwise
    labels, units = [], []
    try:
        for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
            labels.append(str(ch["label"][0]))
            if ch["unit"]:
                units.append(ch["unit"][0])
    except (TypeError, IndexError):  # no channel labels found
        pass
    if not labels:
        labels = [str(n) for n in range(n_chans)]
    if not units:
        units = ["NA" for _ in range(n_chans)]

    # Create MNE object converting from microvolts to volts
    scale = np.array([1e-6 if u == "microvolts" else 1 for u in units])
    info = mne.create_info(ch_names=labels, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(
        (stream["time_series"] * scale).T, info, verbose=False)
    raw.rename_channels(dict(zip(labels, NAMES_1020)))
    raw.set_montage('standard_1020')
    raw.info['bads'] = BAD_CHANNELS
    raw.drop_channels(raw.info['bads'])
    return raw, origin_time


def load_data(xdf_files):
    """ Load data and basic cleaning"""
    participants = {}

    for XDF in xdf_files:
        data, _ = pyxdf.load_xdf(XDF)

        # Find streams
        for stream in data:
            if stream['info']['type'][0] == 'EEG':
                stream_eeg = stream
            elif stream['info']['name'][0] == 'MentalFatigue_Blocks':
                stream_mentalfatigue_blocks = stream
            elif stream['info']['name'][0] == 'MentalFatigue_Math_Task':
                stream_mentalfatigue_math_task = stream
            elif stream['info']['name'][0] == 'MentalFatigue_Questionnaire':
                stream_mentalfatigue_quest = stream

        # Create math task dataframe
        math_tasks_df = stream2df(stream_mentalfatigue_math_task)
        math_task_duration = math_tasks_df['end_ts'] - math_tasks_df['init_ts']
        math_tasks_df['lsl_end_ts'] = stream_mentalfatigue_math_task['time_stamps']
        math_tasks_df['lsl_init_ts'] = math_tasks_df['lsl_end_ts'] - math_task_duration
        math_tasks_df.drop(columns=['init_ts', 'end_ts'], inplace=True)

        # Create blocks dataframe
        blocks_df = stream2df(stream_mentalfatigue_blocks)
        block_duration = blocks_df['end_ts'] - blocks_df['init_ts']
        blocks_df['lsl_end_ts'] = stream_mentalfatigue_blocks['time_stamps']
        blocks_df['lsl_init_ts'] = blocks_df['lsl_end_ts'] - block_duration
        blocks_df.drop(columns=['init_ts', 'end_ts'], inplace=True)

        # Create questionnaire dataframe
        quest_df = stream2df(stream_mentalfatigue_quest)

        # Create MNE object and add annotations
        raw, origin_time = stream2mne(stream_eeg)

        blocks_df['onset'] = blocks_df.lsl_init_ts - origin_time
        blocks_df['duration'] = blocks_df.lsl_end_ts - blocks_df.lsl_init_ts
        blocks_df.sort_values(by='block_idx', inplace=True)

        # MNE Annotations
        annotations = mne.Annotations(blocks_df.onset.values,
                                      blocks_df.duration.values,
                                      blocks_df.block_idx.values.astype(str))
        raw.set_annotations(annotations)

        # Save data
        participants[os.path.splitext(os.path.basename(XDF))[0]] = {"EEG": raw,
                                                                    "MathTasks": math_tasks_df,
                                                                    "Blocks": blocks_df,
                                                                    "Questionnaire": quest_df}

    return participants


def preprocessing_EEG(participants):
    for name, data in participants.items():
        raw = data['EEG']

        # Notch filters (power line and sub-harmonic)
        raw = raw.notch_filter([25, 50], verbose=False)

        # High-pass filter
        raw.filter(2, None, method='iir')

        participants[name]['EEG'] = raw
    return participants


def compute_power(epoch, fmin, fmax):
    psds, freqs = psd_multitaper(epoch,
                                 fmin=fmin,
                                 fmax=fmax,
                                 normalization='full',
                                 verbose=False)
    abs_psds = psds.mean(1).sum(1)
    # absolute power (in dB)
    epoch_absolute_power_db = 10 * np.log10(abs_psds)

    return epoch_absolute_power_db


def compute_relative_power(epoch, fmin, fmax):
    psds, freqs = psd_multitaper(epoch,
                                 fmin=0,
                                 fmax=60,
                                 normalization='full',
                                 verbose=False)

    psds /= np.sum(psds, axis=-1, keepdims=True)
    psds_band = psds[:, :, (freqs >= fmin) & (freqs <= fmax)].sum(axis=-1)
    epoch_relative_power = psds_band.mean(1)

    return epoch_relative_power


def calculate_powers(participants):
    powers = []
    for name, data in participants.items():
        # Get events
        raw = data['EEG']
        events, event_dict = mne.events_from_annotations(raw,
                                                         event_id=EVENTS,
                                                         chunk_duration=1)  # chunks of 1 sec

        # Get epochs by condition and calculate power
        for condition, value in event_dict.items():
            for band, fmin, fmax in FREQ_BANDS:
                epoch_power = {'participant': name,
                               'block': value,
                               'band': band}
                for chs_name, chs in CHANNELS.items():
                    thresholds = []
                    for c in chs:
                        data = raw.get_data(picks=c)
                        mu = np.mean(data)
                        sd = np.std(data)
                        lb, ub = mu-3*sd, mu+3*sd
                        thresholds.append(ub - lb)
                    rejection_threshold = dict(eeg=np.max(thresholds))
                    epoch = mne.Epochs(raw, events, {condition: value}, tmin=0, tmax=1,
                                       picks=chs,
                                       reject=rejection_threshold, baseline=None,
                                       preload=True, verbose=False)
                    if len(epoch) != 0:
                        # Calculate power in different frequency bands
                        power = compute_power(epoch, fmin, fmax)
                        relative_power = compute_relative_power(epoch,
                                                                fmin,
                                                                fmax)
                        epoch_power[chs_name] = power
                        epoch_power[chs_name + '_rel'] = relative_power
                powers.append(epoch_power)

    return pd.DataFrame(powers)


def preprocessing(xdf_files, powers=True, outliers={}):
    participants = load_data(xdf_files)
    participants = preprocessing_EEG(participants)
    if powers:
        powers_df = calculate_powers(participants)
    else:
        powers_df = None
    return participants, powers_df