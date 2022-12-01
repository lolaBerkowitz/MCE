
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
    # read data
    temp = pd.read_csv(file)
    metadata = pd.read_csv(metadata_path)
    basename = os.path.basename(file).split('_')
    video_name = temp.Label[0].split(':')[0]
    meta_idx = metadata["video_file"] == video_name

    # pull metadata fs,pulse_off_idx,
    fs = int(metadata["frame_rate"][meta_idx])
    pulse_off_idx = int(metadata["frame_after_burst"].loc[metadata["video_file"] == video_name])
    pulse_on_idx = int(metadata["frame_before_burst"].loc[metadata["video_file"] == video_name])

    diet = [metadata["diet"][meta_idx].values[0]]*len(temp)
    sex = [metadata["sex"][meta_idx].values[0]]*len(temp)
    

    # compute parameters
    dt = 1/fs
    n_frames = len(temp)


    # create vars 
    name = [temp["Label"][0].split("_")[2]]*len(temp) #subject id
    burst = [temp["Label"][0].split("_")[3]]*len(temp)  # burst id
    time = np.arange(dt,dt*n_frames+dt,dt) # generated timestamps
    pulse_binary = np.linspace(0,n_frames,n_frames) 
    pulse_binary = (pulse_binary <= pulse_on_idx) | (pulse_binary >= pulse_off_idx) # binary indicating frames not contaminated by pulse

    # find burst end (intensity drastically falls beyond 3 SD) 
    # ~~~~~~ NEEDS OPTIMIZATION ACRROSS BACKGROUND NOISE LEVELS ~~~~~~~ LB 01/16
    # smooth_bmode = gaussian_filter(temp.Mean,3)
    # z_diff = stats.zscore(np.diff(temp.Mean))
    # zero_point = np.where(z_diff < -3)[0]+1
    
    # create "time_after_burst" relative to time zero 
    time_from_burst = time - time[pulse_off_idx]

    # create normalize intensity relative to time before pulse
    norm_sig = norm_to_baseline(temp["Mean"],pulse_binary,time_from_burst)

    # estimate cardiac phase from bmode (filters for heart rate frequencys default 5-13Hz)
    # hr, peak_locs, filt_sig = cardiac_cylce_index(time,temp["Mean"])
    
    # build df 
    df = pd.DataFrame()
    df["mouse"] = name
    df['diet'] = diet
    df['sex'] = sex
    df["video_name"] = [temp.Label[0].split(':')[0]]*len(temp)
    df["time"] = time
    df["burst"] = burst
    df["signal"] = temp["Mean"]
    df["time after burst"] = time_from_burst
    df["pulse_binary"] = pulse_binary
    df["norm_signal"] = norm_sig
    # df["phase"] = np.round(np.angle(signal.hilbert(filt_sig)))
    # df["filt_sig"] = filt_sig

    return df 

def norm_to_baseline(sig,pulse_binary,time_after_burst):
    baseline_idx = (pulse_binary == True) & (time_after_burst < 0)
    baseline_mean = np.mean(sig[baseline_idx])
    baseline_sd = np.std(sig[baseline_idx])
    return (sig - baseline_mean) / baseline_sd

def main(data_path,metadata_path,video_path):
    """ process fiji output for echo pipeline
    input: 
        data_path: path to fiji output for bmode intensity of echo image data
        metadata_path: path to metadata csv containing, video frame rate (fs), index 
        of frame after burst(frame_after_burst), and video name (video_file) 

    output:
        
    
    """

    # file_path = "//10.253.5.16/sn data server 3/laura_berkowitz/cardiac_grant/burst_data/bmode_mean_intensity/"
    files = glob.glob(data_path+"*.csv")

    # loop through files
    for file in files: 

        #create subfolder for file
        basename = os.path.basename(file)
        basepath = os.path.join(os.path.dirname(file),basename.split('.')[0])
        # if not os.path.exists(basepath):
        #     os.mkdir(basepath)
        # create dataframe
        df = process_fiji(file,metadata_path)
        # plot phase relative to intensity
        # plot_signal(df,basepath)
        # save images by phase into subfolders 
        # extract_image_by_phase(df,basepath,video_path)
        df.to_csv(os.path.join(basepath,'processed_'+basename))

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


# Plotting functions 

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.
    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def plot_signal(df,save_path):

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

    plt.savefig(save_path+os.sep+"cardiac_cycle_peaks.svg",dpi = 300, bbox_inches = "tight")



