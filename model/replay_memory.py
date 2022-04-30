from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', ))

class replay_memory():
	def __init__(self, cap:int):
		from collections import deque
		from typing import Deque
		self.memory:Deque[Transition]=deque([], maxlen=cap)
	def push(self, state, action, reward, next_state):
		self.memory.append(Transition(state, action, reward, next_state))
	from abc import abstractmethod
	@abstractmethod
	def sample(self, *args, **kwargs)->list:
		pass
	def __len__(self):
		return len(self.memory)

# class scheduled_replay_memory(replay_memory):
# 	from collections import namedtuple
# 	Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'last_time', 'last_loss', ))
# 	def __init__(self, cap:int, window_size:int):
# 		from collections import deque
# 		from typing import Deque
# 		self.memory:Deque[scheduled_replay_memory.Transition]=deque([], maxlen=cap)
# 		self.time = 0
# 		self.window_size = window_size
# 		raise Exception('not completed')
# 	def sample(self, batch_size:int): # 拟实现加窗 HRRN
# 		self.time += batch_size
# 		raise Exception('not completed')
# 	def push(self, state, action, reward, next_state, loss:float):
# 		raise Exception('not completed')

class replay_memory_random(replay_memory):
	def __init__(self, cap:int):
		super().__init__(cap)
	def sample(self, batch_size:int):
		import random
		return random.sample(self.memory, batch_size)

class replay_memory_serial(replay_memory):
	def __init__(self, cap:int):
		super().__init__(cap)
	def sample(self, batch_size:int):
		l = [Transition(None, None, None, None)]*batch_size
		i, j = 0, 0
		while True:
			l[i] = self.memory[j]
			j += 1
			i += 1
			if i>= batch_size: break
			if j==len(self.memory): j = 0
		return l

from typing_extensions import Self
class replay_memory_heap(replay_memory):
	class Transition(Transition):
		from nle_win.basic.core.obs import observation
		def __init__(self, state:observation, action:int, reward:float, next_state:observation, loss:float) -> None:
			super().__init__(state, action, reward, next_state)
			self.loss = loss
		def __lt__(self, rhv): # 将构造大顶堆
			return self.loss > rhv.loss
	def __init__(self, cap:int):
		self.cap = cap
		self.memory = [replay_memory_heap.Transition(None, None, None, None, None)]*0
	def __len__(self):
		return len(self.memory)
	def make_Transition(state, action, reward, next_state, loss:float):
		return replay_memory_heap.Transition(state, action, reward, next_state, loss)
	def push(self, Transition):
		import heapq # 小顶堆
		if len(self.memory) >= self.cap: # 先弹出后压入
			self.memory.pop() # 弹出一个(可能)较大元素
		heapq.heappush(self.memory, Transition)
	def sample(self, batch_size:int):
		return self.memory[0:batch_size]
	def pop(self): # usage: pop, compute loss, modify loss, push
		import heapq
		return heapq.heappop(self.memory)

class replay_memory_windowed_HLR(replay_memory): # Highest Loss Ratio
	class Transition:
		from nle_win.basic.core.obs import observation
		def __init__(self, state:observation, action:int, reward:float, next_state:observation, loss:float) -> None:
			self.data = Transition(state, action, reward, next_state)
			self.loss = loss
			self.time = 0
			self.value = loss
			self.step() # 出于精度考虑单独存储 loss 和 time，出于效率单独存储 value
		def step(self):
			self.time += 1
			from math import log2
			self.value = log2(self.loss/self.time)
		def __lt__(self, o:Self):
			return self.value < o.value
	class Window:
		def __init__(self, cap:int) -> None:
			transition:replay_memory_windowed_HLR.Transition=None
			self.memory = [transition]*cap
			self.next = self
			self.prev = self # 双向循环链表
			# self.smp_next = self # next window to sample。sample 循环链表。
			self.len = 0
		def set_next(self, _next:Self):
			# 使之前的链表相连
			next_ = self.next
			prev_ = self.prev
			next_.prev = prev_
			prev_.next = next_
			# 插入到新链表
			self.next = _next
			self.prev = _next.prev
			self.next.prev = self
			self.prev.next = self
		def push(self, Transition):
			def bisearch(self:replay_memory_windowed_HLR.Window, Transition:replay_memory_windowed_HLR.Transition):
				mem = self.memory # 认为此时的 mem 从大到小排序
				l, r = 0, self.len-1
				if r-l < 1: return l # this algo allows len in [0, 1, 2, ...]
				i = (l+r)>>1
				while l != i:
					if Transition<mem[i]: l = i
					else: r = i
					i = (l+r)>>1
				if Transition<mem[i]: i += 1
				return i
			mem = self.memory # 有序
			n = bisearch(self, Transition)
			if self.len<len(self.memory):
				self.len += 1
			# elif self.smp_next is not self.next: # 满 且 smp_next是链表首，说明 smp 链表可以扩充
			# 	self.next.smp_next = self.smp_next # 下一元素的 smp_next 指向链表首
			# 	self.smp_next = self.next # 扩展 smp 链表
			mem[n+1:self.len] = mem[n:self.len-1] # 从 n 开始后移一格
			mem[n] = Transition # 新元素可能替换掉队尾元素
		def sample(self, batch_size:int):
			if self.len<batch_size:
				transition:replay_memory_windowed_HLR.Transition = None
				batch = [transition]*batch_size
				for i in range(len(batch)):
					batch[i] = self.memory[i%self.len]
			else:
				batch = self.memory[0:batch_size]
				for transition in batch:
					transition.step() # influence batch's and self.memory's elements
				self.memory[0:self.len]=sorted(self.memory[0:self.len], reverse=True) # big->small. doesn't affect batch
			for i, transition in enumerate(batch):
				batch[i] = transition.data
			return batch
		def __len__(self):
			return self.len
		def __getitem__(self, __o):
			return self.memory.__getitem__(__o)
		def __iter__(self):
			return self.memory.__iter__()
		def __contains__(self, __o):
			return self.memory.__contains__(__o)
		def __setitem__(self, *args, **kwargs):
			self.memory.__setitem__(*args, **kwargs)
	def __init__(self, window_size:int, window_num:int):
		'''
		window_size >= batch_size \n
		window_num >= 1 \n
		capacity = window_size * window_num
		'''
		def set_window(window_num:int): # int>=1
			memory = [replay_memory_windowed_HLR.Window(window_size) for _ in range(window_num)]
			if len(memory)>1:
				p = memory[len(memory)-1]
				# n = memory[1%len(memory)]
				t = memory[0]
				p.next = t
				for n in memory[1:]+memory[:1]: # 建立(双向)循环链表
					t.next, t.prev = n, p
					p = t
					t = n
			return memory
		self.memory = set_window(window_num)
		self.w_max_size = window_size
		self.w_max_num = window_num
		self.w_cur = self.memory[0] # current window
		self.len = 0
	def __len__(self):
		return self.len
	def push(self, state, action, reward, next_state, loss:float):
		'''替换掉当前窗口最低 loss率 的行'''
		w_cur = self.w_cur
		Len = self.len - w_cur.len
		w_cur.push(replay_memory_windowed_HLR.Transition(state, action, reward, next_state, loss))
		self.len = Len + w_cur.len
	def sample(self, batch_size:int):
		'''window_size >= batch_size'''
		batch = self.w_cur.sample(batch_size)
		self.w_cur = self.w_cur.next
		return batch