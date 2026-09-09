import json, random
random.seed(42)

with open('datasets/apple_threads.json') as f:
    threads = json.load(f)

sample = random.sample(threads, 100)
for t in sample:
    for turn in t['turns']:
        if turn['role'] == 'customer':
            print(turn['text'])
            break