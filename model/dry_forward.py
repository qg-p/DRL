if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
	del os, sys

from model.misc import *
from model.glyphs import *
from typing import List, Tuple
import torch
import nle_win as nle

def _dry_forward_actions(y:torch.Tensor, inv_55_6:torch.Tensor, misc_6:List[List[int]]):
	no_action_set = [action_set_no(misc) for misc in misc_6]
	a, b, b_inv, c = [], [], [], []
	for misc_i, y_i, inv_i in zip(no_action_set, y, inv_55_6):
		if misc_i==0: l=a
		elif misc_i==1:
			l=b
			b_inv.append(inv_i)
		else: l=c # misc_i==2
		l.append(y_i)
	if len(a): a = torch.stack(a)
	if len(b): b, b_inv = (torch.stack(b), torch.stack(b_inv))
	if len(c): c = torch.stack(c)

	if len(a): a = torch.zeros(*a.shape[:-1], len(actions_list[0]))
	if len(b): b = torch.zeros(*b.shape[:-1], 56)
	if len(c): c = torch.zeros(*c.shape[:-1], len(actions_list[2]))

	y = [a, b, c]
	j = [0, 0, 0]
	Q = [torch.zeros(1)] * len(misc_6)
	for i, misc in enumerate(no_action_set):
		r = y[misc][j[misc]]
		j[misc] += 1
		Q[i] = r
	return Q
def _dry_forward(obs_batch:List[nle.basic.obs.observation], RNN_states:List[Tuple[torch.Tensor, torch.Tensor]]):
	def preprocess(obs_batch:List[nle.basic.obs.observation], device:torch.device):
		from model.glyphs import translate_glyphs, translate_glyph, translate_inv
		default_glyph = 2359 # ' '
		default_translation = translate_glyph(default_glyph)
		batch_size = len(obs_batch)
		map_batch = [None] * batch_size
		surroundings_batch = [None] * batch_size
		blstats_batch = [None] * batch_size
		misc_batch = [[0]] * batch_size
		inv_batch = [None] * batch_size

		for batch_i, obs in enumerate(obs_batch):
			misc_batch[batch_i] = translate_messages_misc(obs)
			inv_batch[batch_i] = translate_inv(obs)
			blstats_batch[batch_i] = obs.blstats
			map_4chnl = translate_glyphs(obs) # map in 4 channels
			map_batch[batch_i] = map_4chnl.tolist()
			srdng5x5_4chnl = [[[int(default_translation[0])]*5]*5, [[int(default_translation[1])]*5]*5, [[int(default_translation[2])]*5]*5, [[int(default_translation[3])]*5]*5]
			y, x = obs.tty_cursor; y -= 1
			for i in range(max(0, y-2), min(map_4chnl.shape[1], y+3)): # shape: (4, 21, 79,)
				for j in range(max(0, x-2), min(map_4chnl.shape[2], x+3)):
					srdng5x5_4chnl[0][i-(y-2)][j-(x-2)] = int(map_4chnl[0][i][j])
					srdng5x5_4chnl[1][i-(y-2)][j-(x-2)] = int(map_4chnl[1][i][j])
					srdng5x5_4chnl[2][i-(y-2)][j-(x-2)] = int(map_4chnl[2][i][j])
					srdng5x5_4chnl[3][i-(y-2)][j-(x-2)] = int(map_4chnl[3][i][j])
			surroundings_batch[batch_i] = srdng5x5_4chnl

		map_batch:torch.Tensor = torch.tensor(map_batch, dtype=torch.float, device=device)
		surroundings_batch:torch.Tensor = torch.tensor(surroundings_batch, dtype=torch.float, device=device)
		blstats_batch:torch.Tensor = torch.tensor(blstats_batch, dtype=torch.float, device=device)
		inv_batch:torch.Tensor = torch.tensor(inv_batch, dtype=torch.float, device=device)
		return map_batch, surroundings_batch, blstats_batch, inv_batch, misc_batch
	device = RNN_states[0][0].device
	(_, _, _, inv_55_6, misc_6) = preprocess(obs_batch, device)
	y = torch.zeros(len(obs_batch), 64)
	Q = _dry_forward_actions(y, inv_55_6, misc_6)
	return Q, RNN_states

def dry_forward_batch( # quick test, check bug
	batch_state:List[nle.basic.obs.observation],
	RNN_STATE:List[List[Tuple[torch.Tensor, torch.Tensor]]],
	models:list
):
	assert len(models)==len(RNN_STATE)
	RETURN_Q:List[List[torch.Tensor]] = [[None]*len(batch_state) for _ in models]
	RETURN_RNN_STATE = [[model.initial_RNN_state()]*len(batch_state) for model in models] # 终止状态的 RNN_STATE 清零
	non_final_mask = [s is not None for s in batch_state]
	if any(non_final_mask):
		INPUT_batch_state = [s for s in batch_state if s is not None] # non final state batch
		INPUT_RNN_STATE = [[rnn_state for (rnn_state, s) in zip(RNN_state, batch_state) if s is not None] for RNN_state in RNN_STATE]

		OUTPUT = [_dry_forward(INPUT_batch_state, rnn_state) for (_, rnn_state) in zip(models, INPUT_RNN_STATE)]
		# RETURN_Q[non_final_mask] = OUTPUT_Q
		j = 0
		for i, non_final in enumerate(non_final_mask):
			if non_final:
				for k, output in enumerate(OUTPUT):
					RETURN_Q[k][i] = output[0][j]
					RETURN_RNN_STATE[k][i] = output[1][j] # ['%016X'%(id(i[0])) for i in RETURN_RNN_STATE[0]]
				j += 1
	return RETURN_Q, RETURN_RNN_STATE
