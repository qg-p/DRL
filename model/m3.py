'''model - autoencoder ver
input:
	observation obs[t]
		obs 21*79
		g'2379' padded obs 5*5
		blstats
	int action
may use LSTM to handle action. require an experiment.

output/predict:
	observation obs[t+1]
		obs 5*5
		blstats
	float reward
'''
'''
此处仅简单试验图像 undercomplete 自动编码器，实现压缩编码。测试要使用的 loss 函数和训练流程。
输入：字符和颜色
输出：与输入相同
为此将实现一个简单数据集。
'''
import torch
from torch import nn

class NetWork(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		pass
	def forward(self):
		pass