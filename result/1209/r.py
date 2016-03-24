import sys

val = []
test = []
with open('e') as f:
    lines = f.readlines()
    v = []
    t = []
    for idx, line in enumerate(lines):
        q = map(float, line.split(' '))
        v.append(q[0])
        t.append(q[1])
        if idx % 10 == 9:
            val.append((sum(v) - max(v))/9.0)
            test.append((sum(t) - max(t))/9.0)
            v = []
            t = []

for item in val:
    print item

print ' '

for item in test:
    print item
