import os
clss = {}

with open('classes.txt') as f:
    classes = f.readlines()

classes = [x.strip() for x in classes]

for i in classes:
    label = i.split(' ')
    clss[label[0]] = label[1]

for img in os.listdir('train/'):
    for lbl in clss:
        if img == lbl:
            os.rename('train/'+img,'train/'+clss[lbl])

for img in os.listdir('val/'):
    for lbl in clss:
        if img == lbl:
            os.rename('val/'+img,'val/'+clss[lbl])