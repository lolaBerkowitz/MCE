import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
import pandas as pd

from scipy import signal 


def plot_signal(df,save_path = None):

    t = np.array(df["time after burst"])
    sig = np.array(df.Mean_bmode)
    filt_sig = df.filt_sig
    peak_locs = signal.find_peaks(filt_sig) 

    fig, axs = plt.subplots(3,1,figsize = (30,10))
    axs = axs.ravel()

    fig.subplots_adjust(hspace = 1,wspace = .5)
    axs[0].plot(t,filt_sig)
    axs[0].scatter(t[peak_locs[0]],filt_sig[peak_locs[0]],c='r')
    axs[0].legend(['filtered signal 5 - 15 Hz','Estimated End-Diastolic Volume'])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Normalized Magnitude')
    axs[0].set_title('End-diastolic volume estimated from peak intensity of filtered ultrasound data')

    axs[1].plot(t,sig)
    # axs[1].scatter(t,sig,c='grey')
    axs[1].scatter(t[peak_locs[0]],sig[peak_locs[0]],c='r')
    axs[1].legend(['raw signal','peaks from filtering'])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Intensity')
    axs[1].set_title('')

    axs[2].plot(t,filt_sig)
    sns.scatterplot(t,filt_sig,hue = df["phase"],palette = "colorblind")
    # axs[2].legend(['filtered signal 5 - 15 Hz','Estimated End-Diastolic Volume'])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Normalized Magnitude')
    axs[2].set_title('Cardiac cycle by phase')

    if save_path is not None:
        plt.savefig(save_path+os.sep+"cardiac_cycle_peaks.svg",dpi = 300, bbox_inches = "tight")


def plot_intensity_by_phase(df):
    """ 
    plots intensity data by phase
    input:
    df: 
    
    
    """
    pw = df[cols].mean(axis=1) 
    t = np.array(df["time after burst"])

    fig= plt.figure(figsize=(20,10))

    sns.relplot(data = df, 
                x = "time after burst",
                y = pw,
                col="phase",
                hue = "phase",
                palette = "colorblind",
                kind = "line")
    sns.relplot(data = df, 
                x = "time after burst",
                y = pw,
                col="phase",
                hue = "phase",
                palette = "colorblind")