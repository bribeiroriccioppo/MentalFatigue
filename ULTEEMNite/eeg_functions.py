# Imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import from_xdf_to_pandas as xp
import matplotlib.pyplot as plt

def create_correlation_df(df, means_questionnaires):
    
    ### Bands df ###
    df_theta = df.loc[(df['band'] == 'Theta')]
    df_theta = df_theta.drop('band', axis=1)
    df_theta = pd.DataFrame(df_theta, dtype = 'float')
    df_alpha = df.loc[(df['band'] == 'Alpha')]
    df_alpha = df_alpha.drop('band', axis=1)
    df_alpha = pd.DataFrame(df_alpha, dtype = 'float')
    df_alpha1 = df.loc[(df['band'] == 'Alpha1')]
    df_alpha1 = df_alpha1.drop('band', axis=1)
    df_alpha1 = pd.DataFrame(df_alpha1, dtype = 'float')
    
    ########################################################

    ### Block means ###

    task_blocks = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
    theta_means = []
    alpha_means = []
    alpha1_means = []

    for block in task_blocks:
        # Theta
        df_theta_block = df_theta.loc[(df_theta['block'] == block)]
        df_theta_block = df_theta_block.drop('block', axis=1)
        theta_block_vec = df_theta_block.to_numpy()
        theta_val = theta_block_vec.mean()
        theta_means.append(theta_val)
        
        # Alpha
        df_alpha_block = df_alpha.loc[(df_alpha['block'] == block)]
        df_alpha_block = df_alpha_block.drop('block', axis=1)
        alpha_block_vec = df_alpha_block.to_numpy()
        alpha_val = alpha_block_vec.mean()
        alpha_means.append(alpha_val)
        
        # Alpha 1
        df_alpha1_block = df_alpha1.loc[(df_alpha1['block'] == block)]
        df_alpha1_block = df_alpha1_block.drop('block', axis=1)
        alpha1_block_vec = df_alpha1_block.to_numpy()
        alpha1_val = alpha1_block_vec.mean()
        alpha1_means.append(alpha1_val)

    ########################################################

    ### Part means ###
    means_per_part_theta = []
    means_per_part_alpha = []
    means_per_part_alpha1 = []

    # 1st 20 minutes:
    theta_means_1 = np.mean(theta_means[0:4])
    alpha_means_1 =  np.mean(alpha_means[0:4])
    alpha1_means_1 =  np.mean(alpha1_means[0:4])
    means_per_part_theta.append(theta_means_1)
    means_per_part_alpha.append(alpha_means_1)
    means_per_part_alpha1.append(alpha1_means_1)

    # 2nd 20 minutes:
    theta_means_2 =  np.mean(theta_means[4:8])
    alpha_means_2 =  np.mean(alpha_means[4:8])
    alpha1_means_2 =  np.mean(alpha1_means[4:8])
    means_per_part_theta.append(theta_means_2)
    means_per_part_alpha.append(alpha_means_2)
    means_per_part_alpha1.append(alpha1_means_2)

    # 3rd 20 minutes:
    theta_means_3 =  np.mean(theta_means[8:12])
    alpha_means_3 =  np.mean(alpha_means[8:12])
    alpha1_means_3 =  np.mean(alpha1_means[8:12])
    means_per_part_theta.append(theta_means_3)
    means_per_part_alpha.append(alpha_means_3)
    means_per_part_alpha1.append(alpha1_means_3)

    ########################################################

    ### Correlation df ###
    df_corr = pd.DataFrame()
    df_corr['Alpha'] = means_per_part_alpha
    df_corr['Alpha 1'] = means_per_part_alpha1
    df_corr['Theta'] = means_per_part_theta
    df_corr['Questionnaire'] = means_questionnaires

    return df_corr

def compute_ratio(participant_df, name):
    # Get participant:

    # Get bands dfs:
    alpha_powers_df = participant_df[(participant_df.band == 'Alpha')] 
    theta_powers_df = participant_df[(participant_df.band == 'Theta')]
    beta_powers_df = participant_df[(participant_df.band == 'Beta')]
    # Reinitialize indexes:
    alpha_powers_df = alpha_powers_df.reset_index(drop=True)
    theta_powers_df = theta_powers_df.reset_index(drop=True)
    beta_powers_df = beta_powers_df.reset_index(drop=True)

    # Get bands values:
    alpha_values = alpha_powers_df['All_rel'].values
    theta_values = theta_powers_df['All_rel'].values
    beta_values = beta_powers_df['All_rel'].values

    # Get ratio values:
    ratio_values = (alpha_values + theta_values)/beta_values
    #sns.boxplot(data=ratio_values)

    # Get ratio df:
    ratio_df = pd.DataFrame(ratio_values, columns=['ratio']) 
    ratio_df['block'] = alpha_powers_df['block'].reset_index(drop=True)

    # Put blocks in the right order:
    ratio_df = ratio_df.sort_values('block',ignore_index=True)
    ordered_ratio_values = ratio_df['ratio'].values
    
    # Plot:
    plt.figure()
    sns.boxplot(data=ordered_ratio_values, showfliers = False).set_title(name)
    plt.savefig(name+'_Ratio.png', dpi=300, bbox_inches='tight')

    return ordered_ratio_values, ratio_df

def compute_ratio_tasks(participant_df, name):
    # Get participant:

    # Get bands dfs:
    alpha_powers_df = participant_df[(participant_df.band == 'Alpha')] 
    theta_powers_df = participant_df[(participant_df.band == 'Theta')]
    beta_powers_df = participant_df[(participant_df.band == 'Beta')]

    # Only task blocks:
    alpha_powers_df = alpha_powers_df[alpha_powers_df.block.isin([13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35])]
    theta_powers_df = theta_powers_df[theta_powers_df.block.isin([13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35])]
    beta_powers_df = beta_powers_df[beta_powers_df.block.isin([13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35])]

    # Reinitialize indexes:
    alpha_powers_df = alpha_powers_df.reset_index(drop=True)
    theta_powers_df = theta_powers_df.reset_index(drop=True)
    beta_powers_df = beta_powers_df.reset_index(drop=True)

    # Get bands values:
    alpha_values = alpha_powers_df['All_rel'].values
    theta_values = theta_powers_df['All_rel'].values
    beta_values = beta_powers_df['All_rel'].values

    # Get ratio values:
    ratio_values = (alpha_values + theta_values)/beta_values
    #sns.boxplot(data=ratio_values)

    # Get ratio df:
    ratio_df = pd.DataFrame(ratio_values, columns=['ratio']) 
    ratio_df['block'] = alpha_powers_df['block'].reset_index(drop=True)

    # Put blocks in the right order:
    ratio_df = ratio_df.sort_values('block',ignore_index=True)
    ordered_ratio_values = ratio_df['ratio'].values
    
    # Plot:
    plt.figure()
    sns.boxplot(data=ordered_ratio_values, showfliers = False).set_title(name)
    plt.savefig(name+'_Ratio_Tasks.png', dpi=300, bbox_inches='tight')

    return ordered_ratio_values, ratio_df

def create_ratio_correlation_df(ratios, means_questionnaires):

    # 1st 20 minutes:
    blocks_ids_1 = [0, 1, 2, 3]
    means_1 = []
    for b in blocks_ids_1:
        mean_block = np.mean(ratios[b])
        means_1.append(mean_block)
    means_1 = np.mean(means_1)

    # 2nd 20 minutes:
    blocks_ids_2 = [4, 5, 6, 7]
    means_2 = []
    for b in blocks_ids_2:
        mean_block = np.mean(ratios[b])
        means_2.append(mean_block)
    means_2 = np.mean(means_2)

    # 3rd 20 minutes:
    blocks_ids_3 = [8, 9, 10, 11]
    means_3 = []
    for b in blocks_ids_3:
        mean_block = np.mean(ratios[b])
        means_3.append(mean_block)
    means_3 = np.mean(means_3)

    means_vec = [means_1, means_2, means_3]
    ########################################################

    ### Correlation df ###
    df_corr = pd.DataFrame()
    df_corr['Ratio'] = means_vec
    df_corr['Questionnaire'] = means_questionnaires

    return df_corr

