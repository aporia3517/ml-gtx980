import sys

val = []
test = []
with open('tanh2layer-v-t') as f:
    lines = f.readlines()
    v = []
    t = []
    for idx, line in enumerate(lines):
        q = map(float, line.split(' '))
        v.append(q[0])
        t.append(q[1])
        if idx % 20 == 19:
            val.append((sum(v) - max(v))/19.0)
            test.append((sum(t) - max(t))/19.0)
            v = []
            t = []

for item in val:
    print item

print ' '

for item in test:
    print item
