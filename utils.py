import time
from typing import Callable

def timeit(func: Callable[..., object]) -> Callable[..., object]:
	def timed(*args: object, **kwargs: object) -> object:
		print(f"Timing function {func.__name__}")
		ts = time.time()
		result = func(*args, **kwargs)
		te = time.time()
		print(f"Function {func.__name__} took {(te-ts)/1000} seconds")
		return result
	return timed
