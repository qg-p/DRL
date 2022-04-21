from .core import obs as OBS, bytes as B
from .common import *

def send_obs(obs:dict)->None:
	OBS.send(obs['glyphs'], obs['chars'], obs['colors'], obs['specials'], obs['blstats'], obs['message'], obs['inv_glyphs'], obs['inv_strs'], obs['inv_letters'], obs['inv_oclasses'], obs['tty_chars'], obs['tty_colors'], obs['tty_cursor'], obs['misc'])

if __name__ == '__main__':
	import nle
	import gym
	env = gym.make('NetHackScore-v0').unwrapped
	obs = env.reset()
	env.close()
	send_obs(obs)