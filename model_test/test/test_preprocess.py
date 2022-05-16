if __name__ == '__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(__file__)+'/../..'))
	del os, sys
import nle_win as nle
from typing import List
import torch

from model_test.explore.glyphs import translate_glyph
default_glyph = 2359 # ' '
default_translation = translate_glyph(default_glyph)

def preprocess(obs_batch:List[nle.basic.obs.observation], device:torch.device):
	from model_test.explore.glyphs import translate_glyphs
	batch_size = len(obs_batch)
	map_batch = [None] * batch_size
	surroundings_batch = [None] * batch_size
	blstats_batch = [None] * batch_size
	misc_batch = [[0]] * batch_size
	inv_batch = [None] * batch_size

	from model_test.explore.glyphs import translate_messages_misc, translate_inv
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

if __name__ == '__main__':
	batch_size = 64
	n_epoch = 4
	use_gpu = True

	device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

	from nle_win.batch_nle.client import connect, disconnect, batch
	connect()
	env = batch(64, 'character="Val-Hum-Fem-Law", savedir=None, penalty_step=-1/64')
	action = [255]*batch_size
	if 1: # warming
		observation = env.step(action)
		obs_batch = [i.obs for i in observation]
		preprocess(obs_batch, device)
	from time import perf_counter
	t = perf_counter()
	for n_ep in range(n_epoch):
		observation = env.step(action)
		obs_batch = [i.obs for i in observation]
		preprocess(obs_batch, device)
	t = perf_counter() - t
	disconnect()
	print(t, batch_size, n_epoch, t/batch_size/n_epoch)