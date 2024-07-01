# import pytracy

# pytracy.set_tracing_mode(pytracy.TracingMode.All)

import sys

import time
def func(*args, **kwargs):
	print(args, kwargs)
	time.sleep(1)

sys.settrace(func)

def g():
	print("G")

while True:
	g()
	print("A")