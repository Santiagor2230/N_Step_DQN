import gym
from gym.wrappers import TransformObservation, NormalizeObservation, \
  NormalizeReward, RecordVideo, RecordEpisodeStatistics, AtariPreprocessing
import numpy as np

def create_environment(name):
  env = gym.make(name)
  env = RecordVideo(env, 'videos', episode_trigger=lambda e: e % 100 == 0)
  env = RecordEpisodeStatistics(env)
  env = AtariPreprocessing(env, frame_skip=8, screen_size=42)
  env = TransformObservation(env, lambda x: x[np.newaxis, :, :])
  env.observation_space = gym.spaces.Box(low=0, high=255, shape=(1,42,42), dtype= np.float32)
  env = NormalizeObservation(env)
  env = NormalizeReward(env)
  return env