def EXEC(cmd:str):
	from nle_win import Exec
	a, b, c, _ = Exec(cmd)
	if not (a and b and c):
		raise Exception()
def test():
	from nle_win import connect, step
	connect()
	EXEC('env.reset()')
	EXEC('env.env.render()')
	obs, reward, done = step(0)
	print(obs, reward, done)
	EXEC('terminate()')
if __name__ == '__main__':
	from nle_win import connect, InteractiveEXEC, disconnect
	connect()
	InteractiveEXEC()
	disconnect()
