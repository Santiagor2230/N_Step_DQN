import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningModule

from experience_replay_buffer import ReplayBuffer

from environment import create_environment

from noisy_dqn_model import DQN



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()

class RLDataset(IterableDataset):

  def __init__(self, buffer, sample_size=400):
    self.buffer = buffer
    self.sample_size = sample_size

  def __iter__(self):
    for experience in self.buffer.sample(self.sample_size):
      yield experience
      
      
      
def greedy(state, net):
  state = torch.tensor([state]).to(device)
  q_values = net(state)
  action = q_values.argmax(dim=-1) #(state, action) dim=-1 is for (action)
  action = int(action.item()) #gives a number
  return action



class DeepQLearning(LightningModule):
  def __init__(self, env_name, policy=greedy, capacity=100_000,
               batch_size=256, lr=1e-3, hidden_size=128, gamma= 0.99,
               loss_fn = F.smooth_l1_loss, optim=AdamW, 
               samples_per_epoch = 1_000,sync_rate=10, a_start=0.5, 
               a_end = 0.0, a_last_episode=100,
               b_start=0.4, b_end=1.0, b_last_episode=100, sigma=0.5, n_steps=3):
    
    super().__init__()
    self.env = create_environment(env_name)

    obs_size = self.env.observation_space.shape
    n_actions = self.env.action_space.n

    self.q_net = DQN(hidden_size, obs_size, n_actions, sigma=sigma) # q network

    self.target_q_net = copy.deepcopy(self.q_net) #target q network

    self.policy = policy
    self.buffer = ReplayBuffer(capacity=capacity)

    self.save_hyperparameters() #saves hyperparameters

    while len(self.buffer) < self.hparams.samples_per_epoch:
      print(f"{len(self.buffer)} samples in experience buffer. Filling...")
      self.play_episode()

  @torch.no_grad()
  def play_episode(self, policy=None):
    state = self.env.reset()
    done = False
    transitions = []

    while not done:
      if policy:
        action = policy(state, self.q_net)
      else:
        action = self.env.action_space.sample()
      next_state, reward, done, info = self.env.step(action)
      exp = (state, action, reward, done, next_state)
      transitions.append(exp)
      state = next_state
    
    for i, (s,a,r,d,ns) in enumerate(transitions):
        batch = transitions[i: i + self.hparams.n_steps]
        # r + gamma * r2 + gamma^2 * r3
        #t[2] = transitions[2] + n_steps of reward trajectory = reward
        # j=0, j=1, j=2 because it is the index
        ret = sum([t[2] * self.hparams.gamma**j for j, t in enumerate(batch)])
        #last done and last state
        _,_,_, ld, ls = batch[-1]
        #state, action, return, last done and last state in the buffer
        self.buffer.append((s,a,ret,ld,ls))

      #forward
  def forward(self, x):
    return self.q_net(x)

  # configure optimizers
  def configure_optimizers(self):
    q_net_optimizer = self.hparams.optim(self.q_net.parameters(), lr=self.hparams.lr)
    return [q_net_optimizer]

  #create dataloader
  def train_dataloader(self):
    dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
    dataloader = DataLoader(
        dataset= dataset,
        batch_size= self.hparams.batch_size
    )
    return dataloader

   #training step
  def training_step(self, batch, batch_idx):
    indices, weights, states, actions, returns, dones, next_states = batch
    weights = weights.unsqueeze(1)
    actions = actions.unsqueeze(1) # creates multiple row in 1 column with unsqueeze(1) (rows = n number of actions, column = 1)
    returns = returns.unsqueeze(1)
    dones = dones.unsqueeze(1)
    
    '''Q = self.q_net(state),  Q = tensor([[0.3,0.4,0.5],[0.1,0.2,0.3]]
    actions = actions.unqueese(1)  actions = tensor([[1],[2]])
    state_action_values = self.q_net(states).gather(1,action)
    state_acton_values = tensor([[0.4],[0.3]])'''
    state_action_values = self.q_net(states).gather(1, actions)

    with torch.no_grad():
      _, next_actions = self.q_net(next_states).max(dim=1, keepdim=True) #array(highest value, location of highest value)
      next_action_values = self.target_q_net(next_states).gather(1,next_actions)
      '''ex: dones = [[false, true, false, true, true]] then the values will only change
      when done = True which is 0.0 it allows for the agent to not take any actions onces
      it reaches the end of the goal'''
      next_action_values[dones] = 0.0 

    expected_state_action_values = returns + self.hparams.gamma**self.hparams.n_steps * next_action_values

    # compute the priorities and update 
    td_errors = (state_action_values - expected_state_action_values).abs().detach()

    for idx, e in zip(indices, td_errors):
      self.buffer.update(idx, e.item())

    #compute the weighted loss function
    loss = weights * self.hparams.loss_fn(state_action_values, expected_state_action_values, reduction='none')
    loss = loss.mean()

    self.log("episode/Q-Error", loss)
    return loss

  #training epoch end
  def training_epoch_end(self, training_step_outputs):

    alpha = max(
        self.hparams.a_end, 
        self.hparams.a_start - self.current_epoch / self.hparams.a_last_episode
    )
    beta = min(
        self.hparams.b_end, 
        self.hparams.b_start + self.current_epoch / self.hparams.b_last_episode
    )

    self.buffer.alpha = alpha
    self.buffer.beta = beta

    self.play_episode(policy=self.policy)
    self.log("episode/Return", self.env.return_queue[-1]) #last episode play by agent

    if self.current_epoch % self.hparams.sync_rate == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())
