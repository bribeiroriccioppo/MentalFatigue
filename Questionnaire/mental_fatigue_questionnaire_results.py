"""
Mental Fatigue Questionnaire
"""

#from scipy import signal
#from skimage.morphology import closing
import os
import numpy as np
import pandas as pd
import statistics as st
import from_xdf_to_pandas as xp
import matplotlib.pyplot as plt

def load_mf_questions(questionnaire):
    # Definitions:
    nb_MF_questions = 11 # Nb of questions related to MF
    nb_questionnaires = 4 # Nb of questionnaires in the experiment
    max_MF_single_score = 6 # Max MF score for a single question
    max_MF_total_score = max_MF_single_score*nb_MF_questions # Max total MF score
    MF_questions = np.zeros((nb_questionnaires, nb_MF_questions))
    MF_means = np.zeros((nb_questionnaires))
    MF_stds = np.zeros((nb_questionnaires))
    # Defining Mental Fatigue Related Questions:
    # Correlated to MF:
    tired = []
    fatigued = []
    drowsy = []
    eyes_open = []
    concentrated = []
    blurred_vision = []
    headache = []
    # Inversely correlated to MF:
    energetic = []
    active = []
    efficient = []
    motivated = []

    for nb in range(nb_questionnaires):
        # Defining Mental Fatigue Related Questions:
        # Correlated to MF:
        tired.append(questionnaire.iloc[nb,1])
        fatigued.append(questionnaire.iloc[nb,2])
        drowsy.append(questionnaire.iloc[nb,3])
        eyes_open.append(questionnaire.iloc[nb,11])
        concentrated.append(questionnaire.iloc[nb,12])
        blurred_vision.append(questionnaire.iloc[nb,13])
        headache.append(questionnaire.iloc[nb,14])

        # Inversely correlated to MF:
        energetic.append(max_MF_single_score-questionnaire.iloc[nb,6])
        active.append(max_MF_single_score-questionnaire.iloc[nb,7])
        efficient.append(max_MF_single_score-questionnaire.iloc[nb,8])
        motivated.append(max_MF_single_score-questionnaire.iloc[nb,9])

        MF_questions_row = [tired[nb], fatigued[nb], drowsy[nb], energetic[nb], active[nb], efficient[nb], motivated[nb], eyes_open[nb], concentrated[nb], blurred_vision[nb], headache[nb]]
        MF_questions[nb] = MF_questions_row

        # Mean & Std calculation:
        MF_means_row = np.mean(MF_questions_row)
        MF_std_row = np.std(MF_questions_row)
        MF_means[nb] = MF_means_row
        MF_stds[nb] = MF_std_row

    return MF_questions, MF_means, MF_stds


def plot_results(questionnaire,name):
    q1 = questionnaire[0,:]
    q2 = questionnaire[1,:]
    q3 = questionnaire[2,:]
    q4 = questionnaire[3,:]
    
    Nb = 11 # nb of questionnaires
    ind = np.arange(Nb) 
    width = 0.2


    plt.figure(figsize =(15, 10))
    bar1 = plt.bar(ind, q1, width, color = 'r')
    bar2 = plt.bar(ind+width, q2, width, color = 'b')
    bar3 = plt.bar(ind+width*2, q3, width, color = 'g')
    bar4 = plt.bar(ind+width*3, q4, width, color = 'c')
    
    plt.xlabel("Questions")
    plt.ylabel('Mental Fatigue Score')
    plt.ylim(0,8)
    plt.xticks(ind+width*1.5,['tired', 'fatigued', 'drowsy', 'energetic', 'active', 'efficient',
          'motivated', 'eyes open', 'concentrated', 'blurred vision', 'headache'])
    plt.legend( (bar1, bar2, bar3, bar4), ('Q1', 'Q2', 'Q3','Q4'))
    plt.savefig(f'C:/Users/bribeiroriccioppo/Desktop/mental-fatigue/Figures/Answers_{name}.jpeg')
    plt.show()

    return


def plot_boxplots(questionnaire, name):
    plt.figure(figsize =(15, 10))
    plt.xlabel("Questionnaires")
    plt.ylabel('Mental Fatigue Score')
    plt.ylim(-0.25,8)
    plt.boxplot(questionnaire, showmeans=True)
    plt.savefig(f'C:/Users/bribeiroriccioppo/Desktop/mental-fatigue/Figures/Boxplot_{name}.jpeg')
    plt.show()

    return

def plot_means(means, name):
    x_axis = [1,2,3,4]
    plt.figure(figsize =(15, 10))
    plt.xlabel("Questionnaires")
    plt.ylabel('Mean Mental Fatigue Score')
    plt.ylim(0,6)
    plt.xticks([1,2,3,4])
    plt.plot(x_axis,means)
    plt.savefig(f'C:/Users/bribeiroriccioppo/Desktop/mental-fatigue/Figures/Means_{name}.jpeg')
    plt.show()
    return
 

if (__name__ == "__main__"):
    questionnaire = xp.load_data(['S1_2022-04-27.xdf'])
    print(questionnaire["S1"]['Questionnaire'])
    questionnaire["S1"]['Questionnaire'].to_csv('Questionnaire.csv')
    results = questionnaire["S1"]['Questionnaire']

