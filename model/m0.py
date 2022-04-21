from ..nle_win import basic
'''
Simple Reinforcement Learning Neural Network Demo to Playing NetHack

# observation 的解释可参照 The NetHack Learning Environment
# 由 github 上的 nle 源代码可知，产生 glyphs 等观测的模块为 pynethack。
# 进一步搜索表明 glyphs 等信息来自 C 语言的 NetHack。
@input&output
输入 1 ：游戏画面 21 * 79 的 char|color、角色坐标、bottom line statistics、inventory glyphs, types, count, BUC
输出 1 ：输入 1 的压缩编码 code 1
输入 2 ：动作 A、A 对应的字母、编码 code 1
输出 2 ：下一画面 21 * 79 的 char|color、角色坐标、blstats、inv glyphs, count, BUC
@model
模仿 Küttler 在 "The NetHack Learning Environment" 中使用的构型，做些许改动。
0 处理输入
1.1 角色周围 5 * 5 的方形区域卷积，输入区域内的地形、光照、怪物、物品（用 char|color 表示）
1.2 21 * 79 地图区域卷积，仅输入地形（墙、各类门（以及方向）、各类陷阱）（分 2 层，第一层为类型，第二层表示是否通行）以及是否光照。压缩映射
1.3 blstat 的全连接 MLP
2 压缩 MLP
3 LSTM
4 MLP
5 输出
@train

@test

'''
class replay_memory:
	basic.obs

import torch
from torch import nn

class Model_0(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.network = nn.Linear(1, 1)
		pass
	def forward(self, S:basic.obs, A:int, R:float, S_:basic.obs, A_:basic.obs, Q:float):
		pass

class Model_1(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.encoder1 = (
			tuple(
				nn.Sequential(
					nn.Conv2d(4, 4, 1),
					nn.ReLU(),
				) for _ in range(3)
			), # resnet block
		)
		self.decoder = ()
		self.predict_net = ()
		self.policy_net = ()
	'''
	@function: encode
	@input: observation (batch)
	@output: obs_code (batch)
	@usage: compress observation
	'''
	def encode(self, observation):
		pass
	'''
	@function: decode
	@input: obs_code (batch)
	@output: observation (batch)
	@usage: optimize encoder
	'''
	def decode(self, obs_code):
		pass
	'''
	@function: predict
	@input: obs_code, action (in batch form)
	@output: next obs_code, Q (batch)
	'''
	def predict(self, obs_code, action):
		pass
	'''
	@function: policy
	@input: obs_code (batch)
	@output: predict reward of every actions (batch)
	'''
	def policy(self, obs_code):
		pass