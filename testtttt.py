import itertools

c = itertools.cycle([0])

for _ in range(5):
    print(next(c))

