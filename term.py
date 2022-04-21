def terminate():
	from nle_win import connect, Exec
	connect()
	a, b, c, _ = Exec('terminate()')
	if not (a and b and c):
		raise Exception('fail to exec')

if __name__ == '__main__':
	terminate()
