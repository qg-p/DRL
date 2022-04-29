class replay_memory():
	from collections import namedtuple
	Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', ))
	def __init__(self, cap:int):
		from collections import deque
		from typing import Deque
		self.memory:Deque[replay_memory.Transition]=deque([], maxlen=cap)
	def push(self, state, action, reward, next_state):
		self.memory.append(replay_memory.Transition(state, action, reward, next_state))
	from abc import abstractmethod
	@abstractmethod
	def sample(self, *args, **kwargs)->list:
		pass
	def __len__(self):
		return len(self.memory)

class scheduled_replay_memory(replay_memory):
	from collections import namedtuple
	Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'last_time', 'last_loss', ))
	def __init__(self, cap: int, window_size: int):
		from collections import deque
		from typing import Deque
		self.memory:Deque[scheduled_replay_memory.Transition]=deque([], maxlen=cap)
		self.time = 0
		self.window_size = window_size
		raise Exception('not completed')
	def sample(self, batch_size:int): # 拟实现加窗 HRRN
		self.time += batch_size
		raise Exception('not completed')
	def push(self, state, action, reward, next_state, loss:float):
		raise Exception('not completed')

class replay_memory_random(replay_memory):
	def __init__(self, cap: int):
		super().__init__(cap)
	def sample(self, batch_size:int):
		import random
		return random.sample(self.memory, batch_size)

class replay_memory_serial(replay_memory):
	def __init__(self, cap: int):
		super().__init__(cap)
	def sample(self, batch_size:int):
		l = [replay_memory.Transition(None, None, None, None)]*batch_size
		i, j = 0, 0
		while True:
			l[i] = self.memory[j]
			j += 1
			i += 1
			if i>= batch_size: break
			if j==len(self.memory): j = 0
		return l