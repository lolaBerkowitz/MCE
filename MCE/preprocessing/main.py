
#data structure and io
import os
import pandas as pd
import cv2
import re 

# general computation 
import numpy as np

# signal processing 
from scipy import signal 
from scipy import stats

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run(file,save = True, metadata_path = r"Y:\laura_berkowitz\cardiac_grant\burst_data\meta_data.csv"):
    """ 
    Main function that processes fiji output, performs rough classification and returns 
    dataframe.
    """
    # process fiji output and returns df 
    df = process_fiji(file,metadata_path)

    # classify observations across phase using kmeans to determine density
    labels_ = classify_phase(df['phase'])

    # Add labels to df 
    df['kmeans_labels'] = labels_
    df['round_labels'] = np.round(df['phase'])
  

    if save:
      df.to_csv(os.path.join(os.path.dirname(file),'processed_'+os.path.basename(file)),index = False)

    return df 

def estimate_pulse(signal,thresh = 3):
  """ 
    Find burst onset and offset from signal (intensity drastically falls beyond 3 SD) 
    ~~~~~~ NEEDS OPTIMIZATION ACRROSS BACKGROUND NOISE LEVELS ~~~~~~~ LB 01/16
    Only works well for data with strong offset associated with pulse
  input: 
    signal: vector of data containing single offset to detect (only handles one pulse)
  output: 
    estimated index for pulse on and pulse off

  To-do: 
    -adapt for signals with multiple pulse on/off 
    -optimize for signals where offset is lower threshold
  """
  z_diff = stats.zscore(np.diff(signal))
  pulse_on = np.where((z_diff > thresh))[0]-1 
  pulse_off = np.where((z_diff < -thresh))[0]-1 

  return min(pulse_on), max(pulse_off)

def filter_signal(ts,sig,lower = 5,upper = 15):
  """ 
  returns bandpass filtered (default 5 - 15hz) signal
  """
  fs = 1/stats.mode(np.diff(ts))[0][0]
  a,b = signal.butter(3,(lower,upper),btype="bandpass",fs = fs)
  return signal.filtfilt(a,b,sig)

def process_fiji(file,metadata_path):
    """ process fiji output for echo pipeline. Creates dataframe used to estimate phase
    inputs: 
        file: path to file containing bmode intensity from fiji. 
        fs: sample rate of image data
        pulse_off_idx: frame number from image data indicating time 0 following ultrasound pulse off. 
    output: 
        df: dataframe containing generated timestamps (time), mouse id (mouse), 
        burst id (burst), mean bmode intensity (Mean_bmode), and timestamps relative 
        to pulse (or burst) event (time_from_burst). 
    
    """    
    # read data file and metadata
    fiji_out = pd.read_csv(file)
    metadata = pd.read_csv(metadata_path)

    meta_idx = metadata["video_file"].str.contains(fiji_out.Label[0].split(':')[0])

    # generate timestamps
    fs = int(metadata["frame_rate"][meta_idx])
    dt = 1/fs
    n_frames = len(fiji_out)
    ts = np.arange(dt,dt*n_frames+dt,dt) # generated timestamps

    # pull metadata fs,pulse_off_idx,
    pulse_off_idx = int(metadata.loc[meta_idx,"frame_after_burst"])
    pulse_on_idx = int(metadata.loc[meta_idx,"frame_before_burst"])

    # create vars 
    pulse_binary = np.linspace(0,len(fiji_out),len(fiji_out)) 
    pulse_binary = (pulse_binary <= pulse_on_idx) | (pulse_binary >= pulse_off_idx) # binary indicating frames not contaminated by pulse
    
    # create "time_after_burst" relative to time zero 
    time_after_burst = ts - ts[pulse_off_idx]

    # create normalize intensity relative to time before pulse
    norm_sig = norm_to_baseline(fiji_out["Mean"],pulse_on_idx)

    # Filter the bmode to so images can be aligned by intensity
    filt_sig = filter_signal(ts,fiji_out["Mean"],lower = 5,upper = 15)
    
    # build df 
    df = pd.DataFrame()
    df["time"] = ts
    df["signal"] = fiji_out["Mean"]
    df["time_after_burst"] = time_after_burst
    df["pulse_binary"] = pulse_binary
    df["norm_signal"] = norm_sig
    df["phase"] = np.round(np.angle(signal.hilbert(filt_sig)),decimals=3)
    df["filt_sig"] = filt_sig
    df["mouse"] = metadata.loc[meta_idx,'mouse_id'][0].values*len(fiji_out)
    df['diet'] = [metadata["diet"][meta_idx].values[0]]*len(fiji_out)
    df['sex'] = [metadata["sex"][meta_idx].values[0]]*len(fiji_out)
    df["video_name"] = [fiji_out.Label[0].split(':')[0]]*len(fiji_out)
    df["burst"] = [fiji_out["Label"][0].split("_")[3]]*len(fiji_out)
    
    return df 

def norm_to_baseline(sig,pulse_on_idx):
    baseline_mean = np.mean(sig[0:pulse_on_idx])
    baseline_sd = np.std(sig[0:pulse_on_idx])
    return (sig - baseline_mean) / baseline_sd

def classify_phase(phase,n_clusters = 6,random_state = 42, max_iter = 300, n_init = 20):
    # performs k-means clustering on phase to estimate density of 
    # intensity values across phase. 
    # input 
    features = np.vstack([phase]).T
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(
        init="random",
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )

    kmeans.fit(scaled_features)

    return kmeans.labels_

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