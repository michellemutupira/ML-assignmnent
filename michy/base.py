import streamlit as st
import os
from PIL import Image

vid_path = './car.mp4'
def play_video(vid_path):
    try:
        video_file = open(vid_path, 'rb')
        video_bytes = video_file.read()
        st.video(vid_path)
    except FileNotFoundError:
        pass
