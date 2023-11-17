from IPython.display import HTML
from base64 import b64encode

def display_video(episode=0):
  video_file = open(f'/videos/rl-video-episode-{episode}.mp4', "r+b").read()
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"<video width=600 controls><source src='{video_url}'></video>")

display_video(episode=0)