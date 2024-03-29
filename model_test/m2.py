# Hierarchical Deep Q-Learning
'''
解决稀疏奖励问题

分为上层长程决策和下层行动
上层：
输出
是否：移动、攻击、使用、投掷、饮用、穿上、脱下、食用、读卷轴、读书、掉落、拾取、付款
对应：坐标、方向、物品、物品、物品、物品、物品、物品、物品、物品、物品和数量、无、无
的 Q
环境输入
全局的地图、blstats、inv
奖励输入
Delta 的 Q

注
移动到的坐标实际上是取输出 1(C)*21(H)*79(W) 的最大值的下标
攻击方向也是取 8 个值的最大值

下层：
环境输入
每回合的：
周边地图环境(ViT?)、blstats、message 的 encoder-decoder 压缩编码
奖励输入
每步 Rt-ε，到达目的完成则 +ε*最小步数，未到达目的且完成不加分。完成时唤醒上层网络。
输出

'''
