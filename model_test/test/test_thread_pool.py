from concurrent.futures import ThreadPoolExecutor

def func(i, no):
	print(f"func #{(no)} start")
	print(i)
	print(f"func #{(no)} end")
	return no

thread_pool_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="test_")

lst = bytes(range(ord('a'), ord('z')+1)).decode()

futures = [thread_pool_executor.submit(func, lst[i*2:i*2+6], i+1) for i in range(10)]

results = [future.result() for future in futures]
print(results)
# thread_pool_executor.shutdown(wait=True)