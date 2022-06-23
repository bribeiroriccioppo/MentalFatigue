# Copyright (c) 2022
# Manuel Cherep <mcherep@logitech.com>

"""
EEG pre-processing of the data.
"""

from mne.time_frequency import psd_multitaper
import neurodsp
import mne
import numpy as np
import os
import pyxdf
import pandas as pd

NAMES_1020 = ['C4', 'C2', 'Cz', 'C1', 'C3', 'FT8', 'TP8', 'FT7', 'TP7', 'Fp1', 'Fp2']
BAD_CHANNELS = ['C2', 'Cz', 'C1', 'C3', 'C4', 'FT8', 'FT7', 'TP7', 'TP8'
                'F8', 'F3', 'F4', 'P4', 'P3', 'T3', 'T4', 'Fp1']  # Unused channels

CHANNELS = {'All': ['Fp1'],
            'Fp1': ['Fp1']}

#CHANNELS = {'All': ['Channel'],
#            'Channel': ['Channel']}

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
    df = {}
    data = []
    participant = {}

    ##### Data from headband #####

    for CSV in csv_files:
        data_file = pd.read_csv(CSV, index_col=None, header=0)
        data.append(data_file)

    df = pd.concat(data, axis=0, ignore_index=True) 

    # Converting to mV
    df[' EEG'] = df[' EEG']*(106 / (160 * 2^17)) 
    df[' EEG[-]'] = df[' EEG[-]']*(106 / (160 * 2^17))

    """ # Notch filters (sub-harmonic and power line)
    mne.filter.notch_filter(df[' EEG'],256,[25,50])
    mne.filter.notch_filter(df[' EEG[-]'],256,[25,50])

    # High-pass filter
    mne.filter.filter_data(df[' EEG'],256,2,None,method='iir')
    mne.filter.filter_data(df[' EEG[-]'],256,2,None,method='iir') """

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
    #blocks_df.drop(columns=['init_ts', 'end_ts'], inplace=True)


    ##### Creating MNE object #####
    ch_names = ['Channel']
    fs = 256
    origin_time = df[' TimeStamp']
    
    df_T = (df).T

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(df_T.iloc[2:3], info, verbose=False)

    raw = raw.notch_filter([25, 50], verbose=False)
    raw.filter(2, None, method='iir')

    # MNE Annotations
    blocks_df['onset'] = blocks_df.lsl_init_ts - origin_time
    blocks_df['duration'] = blocks_df.lsl_end_ts - blocks_df.lsl_init_ts
    blocks_df.sort_values(by='block_idx', inplace=True)

    annotations = mne.Annotations(blocks_df.onset.values,
                                    blocks_df.duration.values,
                                    blocks_df.block_idx.values.astype(str))

    print(annotations)
    raw.set_annotations(annotations)

    
    participant[os.path.splitext(os.path.basename(XDF))[0]] = {"EEG": raw,
                                                        "Blocks": blocks_df}


    return participant, df, origin_time


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


def calculate_powers(participant):
    powers = []
    for name, data in participant.items():
        # Get events
        raw = data['EEG']

        events, event_dict = mne.events_from_annotations(raw,
                                                         event_id=EVENTS,
                                                         chunk_duration=1)  # chunks of 1 sec

        print(events)
        print(event_dict)

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
                    epoch = mne.Epochs(raw, events, {condition: value}, tmin=0, tmax=1, # 1 minute epoch
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


def get_power(filtered_eeg, win, nb_rec, resp_start, rej_th, FREQ_BANDS, time):

    df_power = pd.DataFrame()
    df_EEG = pd.DataFrame()

    for i in range(0, nb_rec-win, win):
        # get filtered eeg data 
        EEG_Ch = filtered_eeg[i:i + win]
        bad_Ch = 0
        if np.max(EEG_Ch)-np.min(EEG_Ch) > rej_th:
            bad_Ch = 1  
        
        blc_id = 0
        # get block id 
        if time[i]>= resp_start:
            blc_id=1
        # get poower
        pow_Ch = []
        p_sum_Ch = 0
        rel_pow_data_Ch = []
    
        for f in FREQ_BANDS:
            
            freqs_Ch, powers_Ch = compute_spectrum_welch(EEG_Ch, 125, f_range=[f[1], f[2]])
            pow_Ch.append(np.mean(powers_Ch))
            p_sum_Ch = p_sum_Ch + np.mean(powers_Ch)
            
        rel_pow_data_Ch = (pow_Ch/p_sum_Ch)
    
    
        # make eeg df 
        df_eeg = pd.DataFrame()
        df_eeg["EEG_Ch"] = EEG_Ch
        df_eeg["bad_Ch"] = [bad_Ch]*len(EEG_Ch)
        df_eeg["block"] = [blc_id]*len(df_eeg)
        df_eeg["time"] = time[i:i+win]
        df_EEG = pd.concat([df_EEG, df_eeg])
    
        # make pow df
        df_pow_Ch = pd.DataFrame([pow_Ch], columns=["Delta_Ch","Theta_Ch","Alpha_Ch","Beta_Ch","Gamma_Ch"])
        df_pow_Ch_rel = pd.DataFrame([rel_pow_data_Ch], columns=["Delta_Ch_rel","Theta_Ch_rel","Alpha_Ch_rel","Beta_Ch_rel","Gamma_Ch_rel"])
        
        df_pow = pd.concat([df_pow_Ch_rel ,df_pow_Ch], axis=1)
        df_pow["time"]=time[i]
        df_pow["bad_Ch"]=bad_Ch
        df_power = pd.concat([df_power, df_pow])
    
    df_power = df_power.reset_index(drop=True)
    df_EEG = df_EEG.reset_index(drop=True)

    return df_power, df_EEG




def preprocessing(csv_files, XDF, powers=True, outliers={}):
    participant, raw_mne, times = load_data(csv_files, XDF)
    if powers:
        powers_df = calculate_powers(raw_mne)
    else:
        powers_df = None
    return participant, powers_df