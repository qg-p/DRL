if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/../../..'))
	del os, sys
from nle_win.basic.core.obs.observation import observation

class replay_memory_WHLR: # Highest Loss Ratio
	'''每个窗口 Loss 率最大的前 batch_size 个时间长度为 queue_len 的 transition 序列'''
	class Transition:
		'''
		A|R|S, loss, time
		'''
		from nle_win.basic.core.obs import observation
		class Transition_data:
			from nle_win.basic.core.obs import observation
			def __init__(self, action_index:int, reward:float, next_state:observation) -> None:
				self.action_index = action_index
				self.reward = reward
				self.state = next_state
		def __init__(self, state:observation, action_index:int, reward:float, next_state:observation, loss:float) -> None:
			self.data = replay_memory_WHLR.Transition.Transition_data(action_index, reward, next_state)
			self.value = loss
		def step(self, loss:float):
			self.value = loss
		from typing_extensions import Self
		def __lt__(self, o:Self):
			return self.value < o.value
	class Transition_q(Transition):
		'''
		Transition queue:
			Transition 加上一段历史以支持 RNN STATE 的计算
		'''
		from typing_extensions import Self
		from nle_win.basic.core.obs import observation
		def __init__(self, state:observation, action_index:int, reward:float, next_state:observation, loss:float, last_Transition_q:Self, queue_len:int) -> None:
			'if state is None: {.queue=[..., None, state], .data={action_index, reward, next_state}}'
			super().__init__(state, action_index, reward, next_state, loss)
			self.queue = [state]
			if state is not None: # 上一状态不是 None，则 replay_memory_WHLR.sample 已被调用，last_Transition_q is not None。
				prev_queue = last_Transition_q.queue[1:]
			else:
				state:observation = None
				prev_queue = [state]*(queue_len-1) # [None, ...]
			self.queue = prev_queue + self.queue # array: observation[len_queue]
	class Window:
		def __init__(self, cap:int) -> None:
			transition_q:replay_memory_WHLR.Transition_q=None
			self.memory = [transition_q]*cap
			self.next = self
			self.prev = self # 双向循环链表
			# self.smp_next = self # next window to sample。sample 循环链表。
			self.len = 0
		from typing_extensions import Self
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
		def push(self, Transition_q):
			def bisearch(self:replay_memory_WHLR.Window, Transition_q:replay_memory_WHLR.Transition_q):
				mem = self.memory # 认为此时的 mem 从大到小排序
				l, r = 0, self.len-1
				if r-l < 1: return l # this algo allows len in [0, 1, 2, ...]
				i = (l+r)>>1
				while l != i:
					if Transition_q<mem[i]: l = i
					else: r = i
					i = (l+r)>>1
				if Transition_q<mem[i]: i += 1
				return i
			mem = self.memory # 有序
			n = bisearch(self, Transition_q)
			if self.len<len(self.memory):
				self.len += 1
			# elif self.smp_next is not self.next: # 满 且 smp_next是链表首，说明 smp 链表可以扩充
			# 	self.next.smp_next = self.smp_next # 下一元素的 smp_next 指向链表首
			# 	self.smp_next = self.next # 扩展 smp 链表
			mem[n+1:self.len] = mem[n:self.len-1] # 从 n 开始后移一格
			mem[n] = Transition_q # 新元素可能替换掉队尾元素
		def sample(self, batch_size:int):
			if self.len<batch_size:
				transition_q:replay_memory_WHLR.Transition_q = None
				batch = [transition_q]*batch_size
				for i in range(len(batch)):
					batch[i] = self.memory[i%self.len]
			else:
				self.memory[0:self.len]=sorted(self.memory[0:self.len], reverse=True) # big->small. doesn't affect batch
				batch = self.memory[0:batch_size]
				# for transition_q in batch:
				# 	transition_q.step() # influence batch's and self.memory's elements
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
	def __init__(self, window_size:int, window_num:int, queue_len:int):
		'''
		window_size >= batch_size \n
		window_num >= 1 \n
		capacity = window_size * window_num \n
		queue_len >= 1
		'''
		def set_window(window_num:int): # int>=1
			memory = [replay_memory_WHLR.Window(window_size) for _ in range(window_num)]
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
		self.queue_len = queue_len
		self.memory = set_window(window_num)
		self.w_max_size = window_size
		self.w_max_num = window_num
		self.w_cur = self.memory[0] # current window
		self.len = 0
	def __len__(self):
		return self.len
	def __str__(self):
		return f'{(type(self).__name__)}(window_size={(len(self.memory[0].memory))}, window_num={(len(self.memory))}, queue_len={(self.queue_len)})'
	def push(self, state, action_index, reward, next_state, loss:float, transition_q:Transition_q):
		'''
		替换掉当前窗口最低 loss 率的行，返回新的 transition_q。
		没有 transition_q 可以用 None state
		'''
		transition_q = replay_memory_WHLR.Transition_q(state, action_index, reward, next_state, loss, transition_q, self.queue_len)
		w_cur = self.w_cur
		Len = self.len - w_cur.len
		w_cur.push(transition_q)
		self.len = Len + w_cur.len
		return transition_q
	def sample(self, batch_size:int):
		'''
		window_size >= batch_size
		return: array transition[batch_size][(<=)queue_len]
		'''
		batch = self.w_cur.sample(batch_size)
		self.w_cur = self.w_cur.next
		return batch