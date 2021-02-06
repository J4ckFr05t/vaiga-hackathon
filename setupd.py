import os
import shutil
from progressbar import ProgressBar
pbar = ProgressBar()
pbar1 = ProgressBar()
pbar2 = ProgressBar()

unlabelled = []
#read file names
unlabelled = os.listdir('images/')
unlabelled.sort()

train = {}
test = {}
val = {}

#train
with open('train.txt') as f:
    content = f.readlines()

content = [x.strip() for x in content]

#val
with open('val.txt') as f:
    content1 = f.readlines()

content1 = [x.strip() for x in content1]

#test
with open('test.txt') as f:
    content2 = f.readlines()

content2 = [x.strip() for x in content2]



#filename gives label : train
for i in content:
    label = i.split(' ')
    train[label[0]] = label[1]


#filename gives label : val
for i in content1:
    label = i.split(' ')
    val[label[0]] = label[1]

#filename gives label : test
for i in content2:
    label = i.split(' ')
    test[label[0]] = label[1]

#create folders with name
#0 to 101
try:
    for i in range(0,102):
        try:
            os.mkdir('val/'+str(i))
            os.mkdir('train/'+str(i))
        except:
            print('directory exits')
            break
except:
    print("something went wrong")


#train
for img in pbar(unlabelled):
    for lbl in train:
        if img == lbl:
            shutil.copy2('images/'+img, 'train/'+train[lbl]+'/')

print('created train :)')

#val
for img in pbar1(unlabelled):
    for lbl in val:
        if img == lbl:
            shutil.copy2('images/'+img, 'val/'+val[lbl]+'/')

print('created validation :)')

#test
for img in pbar2(unlabelled):
    for lbl in test:
        if img == lbl:
            shutil.copy2('images/'+img, 'test/')

print('created test :)')








