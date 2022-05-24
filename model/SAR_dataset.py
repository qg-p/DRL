if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
	del os, sys
class SAR_dataset:
	class SARI:
		from nle_win.basic.core.obs.observation import observation
		def __init__(self, state:observation, action:int, reward:float, action_index:int) -> None:
			self.state = state
			self.action = action
			self.reward = reward
			self.action_index = action_index

	def __init__(self, filename:str, is_xz:bool=None):
		if is_xz is None:
			import os
			is_xz = os.path.splitext(filename)[1] in ['.xz', '.xzip']
		from dataset.dataset import ARS_dataset_xz, ARS_dataset
		dataset = ARS_dataset_xz(filename, keep_exist_data=True) if is_xz else ARS_dataset(filename, keep_exist_data=True)
		dataset = dataset.readall()

		None_state:SAR_dataset.SARI.observation = None
		self.dataset = [SAR_dataset.SARI(state=None_state, action=255, reward=0., action_index=None)]*0

		from model.memory_replay.dataset.replay_train import index_of_action # 根据 action 反求 action_index
		state = None
		for THIS, NEXT in zip(dataset[:-1], dataset[1:]):
			action, reward = THIS.action, THIS.reward
			action_index = index_of_action(state, action)
			state = THIS.state if NEXT.action != 255 else None
			self.dataset.append(SAR_dataset.SARI(state=state, action=action, reward=reward, action_index=action_index))
		assert self.dataset[-1].state is None # dataset 以 None 结束（且长度至少为一）

		self.iter = iter(self.dataset)

	def next(self)->SARI:
		try:
			return next(self.iter)
		except StopIteration:
			self.iter = iter(self.dataset)
			return self.next()

	def __getitem__(self, *args, **kwargs):
		return self.dataset.__getitem__(*args, **kwargs)
