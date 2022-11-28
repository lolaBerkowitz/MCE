# import dependencies

#data structure and io
import glob
import os
import pandas as pd

# general computation 
import numpy as np

# plotting
import matplotlib.pyplot as plt 

# signal processing 
from scipy.ndimage import gaussian_filter,gaussian_filter1d
from scipy import signal 
from scipy.fft import fft, fftfreq
from scipy import stats


def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def cardiac_cylce_index(ts,sig,lower = 5,upper = 15):
    """ finds indicies for estimated end-diastolic volume 
    by first filtering the signal for heart rate followed by 
    find_peaks to estimate start and end. 

    input: 
    ts: time in seconds
    sig: signal with clear oscillatory comonent 
    lower: lower frequency for filtering
    upper: upper frequency for filtering 
    
    output: 
    hr: estimated heart rate in Hz
    peak_locs: index of end-diastolic volume """

    fs = 1/stats.mode(np.diff(ts))[0][0]
    a,b = signal.butter(3,(5,15),btype="bandpass",fs = 48)
    filt_sig = signal.filtfilt(a,b,sig)
    peak_locs = signal.find_peaks(filt_sig)
    hr = len(ts[peak_locs[0]])/(ts[-1] - ts[0])
    return hr, peak_locs,filt_sig

def extract_image_by_phase(df,basepath,video_path):

  cap = cv2.VideoCapture(video_path+os.sep+df.video_name.iloc[0])

  frame_number = df.index[df['time after burst'] == 0].values[0]
  while cap.isOpened():
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
    res, frame = cap.read()
    if frame_number > len(df)-1:
      break

    file_name = "phase"+str(df["phase"][frame_number])+"frame_"+str(frame_number)+".tiff"
    save_folder = os.path.join(basepath,str(df["phase"].iloc[frame_number]))
    if not os.path.exists(save_folder):
      os.mkdir(save_folder)
    cv2.imwrite(os.path.join(save_folder,file_name), frame)     # save frame as JPEG file      
    # print('Read a new frame: ', success)
    frame_number += 1

  cap.release()