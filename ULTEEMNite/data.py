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

FREQ_BANDS = [('Delta', 1, 3),
              ('Theta', 4, 7),
              ('Alpha1', 8, 10),
              ('Alpha2', 11, 13),
              ('Alpha', 8, 13),
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

def load_data(csv_files, XDF):
    """ Load data and basic cleaning"""
    participants = {}
    data = []


    ##### Data from headband #####

    for CSV in csv_files:
        data_file = pd.read_csv(CSV, index_col=None, header=0)
        data.append(data_file)

    participants = pd.concat(data, axis=0, ignore_index=True) 

    # Converting to mV
    participants[' EEG'] = participants[' EEG']*(106 / (160 * 2^17)) 
    participants[' EEG[-]'] = participants[' EEG[-]']*(106 / (160 * 2^17))

    """ # Notch filters (sub-harmonic and power line)
    mne.filter.notch_filter(participants[' EEG'],256,[25,50])
    mne.filter.notch_filter(participants[' EEG[-]'],256,[25,50])

    # High-pass filter
    mne.filter.filter_data(participants[' EEG'],256,2,None,method='iir')
    mne.filter.filter_data(participants[' EEG[-]'],256,2,None,method='iir') """

    ##### Data from LSL #####

    data_xdf, _ = pyxdf.load_xdf(XDF)

    # Find streams
    for stream in data_xdf:
        if stream['info']['name'][0] == 'MentalFatigue_Blocks':
            stream_mentalfatigue_blocks = stream

    # Create blocks dataframe
    blocks_df = stream2df(stream_mentalfatigue_blocks)
    block_duration = blocks_df['end_ts'] - blocks_df['init_ts']
    blocks_df['lsl_end_ts'] = stream_mentalfatigue_blocks['time_stamps']
    blocks_df['lsl_init_ts'] = blocks_df['lsl_end_ts'] - block_duration
    blocks_df.drop(columns=['init_ts', 'end_ts'], inplace=True)


    ##### Creating MNE object #####
    ch_names = ['Channel']
    fs = 256
    origin_time = participants[' TimeStamp']
    
    participant_T = (participants).T

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(participant_T.iloc[2:3], info, verbose=False)

    raw = raw.notch_filter([25, 50], verbose=False)
    raw.filter(2, None, method='iir')

    # MNE Annotations
    blocks_df['onset'] = blocks_df.lsl_init_ts - origin_time
    blocks_df['duration'] = blocks_df.lsl_end_ts - blocks_df.lsl_init_ts
    blocks_df.sort_values(by='block_idx', inplace=True)

    annotations = mne.Annotations(blocks_df.onset.values,
                                    blocks_df.duration.values,
                                    blocks_df.block_idx.values.astype(str))
    raw.set_annotations(annotations)

    participants[os.path.splitext(os.path.basename(XDF))[0]] = {"EEG": raw,
                                                                "Blocks": blocks_df}


    return participants, raw, origin_time


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


def calculate_powers(raw):
    powers = []

    # Set epoching parameters
    event_id, tmin, tmax = 1, 0, 1
    baseline = None

    # Extract events
    #events = mne.find_events(raw, stim_channel=None)
    events = mne.events_from_annotations(raw, event_id=EVENTS, chunk_duration=1)  # chunks of 1 sec
    print(events)


    for band, fmin, fmax in FREQ_BANDS:
        epoch_power = {'band': band}

        # (re)load the data to save memory
        #raw.pick_types(meg='grad', eog=True)  # we just look at gradiometers
        #raw.load_data()

        # Epoching
        epoch = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                            reject=dict(grad=4000e-13, eeg=40e-3), # eeg in mV
                            preload=True)
        if len(epoch) != 0:
            # Calculate power in different frequency bands
            power = compute_power(epoch, fmin, fmax)
            relative_power = compute_relative_power(epoch,
                                                    fmin,
                                                    fmax)
            epoch_power['Ch'] = power
            epoch_power['Ch_rel'] = relative_power

        powers.append(epoch_power)
       
    
    return pd.DataFrame(powers)


def preprocessing(csv_files, XDF, powers=True, outliers={}):
    participant, raw_mne, times = load_data(csv_files, XDF)
    if powers:
        powers_df = calculate_powers(raw_mne)
    else:
        powers_df = None
    return participant, powers_df