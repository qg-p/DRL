class replay_memory():
	from collections import namedtuple
	Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', ))
	def __init__(self, cap:int):
		from collections import deque
		from typing import Deque
		self.memory:Deque[replay_memory.Transition]=deque([], maxlen=cap)
	def push(self, state, action, reward, next_state):
		self.memory.append(replay_memory.Transition(state, action, reward, next_state))
	def sample(self, batch_size:int):
		import random
		return random.sample(self.memory, batch_size)
	def __len__(self):
		return len(self.memory)
		