# N_Step_DQN
N_Step Noisy Double DQN with Prioritized Experience Replay 

# Requirements
gym[atari,accept-rom-license] == 0.23.1

pytorch-lightining == 1.6.0

stable-baseline3

torch == 2.0.1

# Collab installations
!apt-get install -y xvfb

!pip install \
  gym[atari,accept-rom-license]==0.23.1 \
  pytorch-lightning==1.6.0 \
  stable-baselines3 \
  pyvirtualdisplay


# Description
N_step DQN is a continuation of Noisy DQN, in this implementation we adjust the environment by allowing the agent to see more information of the environment by stacking frames in a single state as well as we adjusted the rewards the agent gets by allowing rewards to be adjusted based on number of steps instead of immmediate reward, this allows us to predict even further in the future which can allow the neural network to optimize better.

# Game
Pong

# Architecture
N_step Noisy Double DQN with Priotirized Experience Replay

# optimizer
AdamW

# Loss
smooth L1 loss function

# Video Results

https://github.com/Santiagor2230/N_Step_DQN/assets/52907423/8afdc826-f92f-47e0-bbba-28e9b9fabee4

