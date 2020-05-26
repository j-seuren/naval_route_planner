import pickle
import random

with open('output\population01', 'rb') as file:
    parents = pickle.load(file)

for i in range(50):
    print(random.random())