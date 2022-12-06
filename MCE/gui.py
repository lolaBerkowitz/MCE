import csv
from functools import partial
import tkinter as tk
from tkinter import filedialog, RepeatingTimer
import cv2
from pathlib import Path
import os

import napari
from napari.utils.notifications import show_info

import tkinter as tk
import cv2

# Create the main window
window = tk.Tk()

# Create a frame to hold the video frames
video_frame = tk.Frame(window)


# Read in the video file
video = cv2.VideoCapture(r'Y:\laura_berkowitz\cardiac_grant\video_data\__713_burst3_48-2021-10-27-00-30-09_1.avi')

# Function to update the video frame
def update_frame():
    # Grab the next frame from the video capture object
    ret, frame = video.read()
    
    # If there are no more frames in the video, stop the timer
    while(ret == True):
        # Convert the frame to a format that Tkinter can display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)
        
        # Update the frame in the GUI
        video_frame.configure(image=frame)
        video_frame.image = frame

# Create a timer to call the update_frame function


# Add buttons to control the playback of the video
play_button = tk.Button(window, text="Play")
play_button.after(100, update_frame)
pause_button = tk.Button(window, text="Pause", command=timer.stop)
play_button.after(100, update_frame)
# Add text fields where the user can enter notes about each frame
note_label = tk.Label(window, text="Notes:")
note_field = tk.Text(window)

# Arrange the widgets in the window
video_frame.pack()
play_button.pack()
pause_button.pack()
note_label.pack()
note_field.pack()

# # this is where data will be written
# currdir = os.getcwd()
# tempdir = filedialog.askdirectory(parent= tk.Tk(), initialdir=currdir, title='Please select a directory')
# basename = os.path.basename(tempdir)
# save_path = os.path.join(tempdir, basename +'_cardiac_phase.csv')

# CSV_OUT = Path(save_path)
# if not CSV_OUT.exists():
#     CSV_OUT.write_text("file,frame,object,action\n")

# # adjust keybindings to your liking
# KEYMAP = {

# }

# viewer = napari.Viewer()

# # this writes the frame, layer source, and action each time you press a key
# def on_keypress(key, viewer):
#     if (key == 'd') | (key == 'f'):
#         object = '1'
#     elif (key == 'j') | (key == 'k'):
#         object = '2'
#     if (key == 'd') | (key == 'j'):
#         action = 'start'
#     elif (key == 'f') | (key == 'k'):
#         action = 'stop'
#     frame = viewer.dims.current_step[0]
#     layer = viewer.layers.selection.active or viewer.layers[-1]

#     show_info('object :'+object+' '+action+' Frame: '+str(frame))  # if you want some visual feedback
#     with open(CSV_OUT, 'a') as f:
#         csv.writer(f).writerow([layer.source.path, frame, object, action])

# for key in KEYMAP:
#     viewer.bind_key(key, partial(on_keypress, key))

# napari.run()

