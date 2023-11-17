from collections import deque, namedtuple
import numpy as np
import random

class ReplayBuffer:

  # constructor.
  def __init__(self, capacity):
    self.buffer = deque(maxlen= capacity)
    self.priorities = deque(maxlen=capacity)
    self.capacity = capacity
    '''alpha = 1.0 we select the sample experience prority to once with highest potential
    when alpha = 0.0 all probabilities of each experience are chosen for
    learning regardles of priority we decrease alpha by time'''
    self.alpha = 1.0 

    '''when emphize sample with highest potential beta allows to avoid
    bias'''
    self.beta = 0.5
    #max priority holds the highest value in priority list
    self.max_priority = 0.0


  # __len__
  def __len__(self):
    return len(self.buffer)

  # Append.
  def append(self, experience):
    self.buffer.append(experience)
    self.priorities.append(self.max_priority)

  # Update.
  def update(self, index, priority):
    if priority > self.max_priority:
      self.max_priority = priority
    self.priorities[index] = priority

  # Sample.
  #sample size for training
  def sample(self, batch_size):
    prios = np.array(self.priorities, dtype=np.float64) + 1e-4
    prios = prios ** self.alpha #p(i)**a
    probs = prios / prios.sum() #p(i)**a/ sum of all priorities

    weights = (self.__len__()* probs) ** -self.beta
    weights = weights/ weights.max() #(1/N * 1/prob(i))**-B

    '''idx will show the random choice of each probability of prior'''
    idx = random.choices(range(self.__len__()), weights = probs, k=batch_size)

    '''we insert the index, the weight of that index and the rest which is:
    [state, reward, action , info] in *self.buffer[i]'''
    sample = [(i, weights[i], *self.buffer[i]) for i in idx]
    return sample