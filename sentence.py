from  settings  import *
def getsen (s = 'flurried'):
    d = open('data/allsen')
    #w = open(s,'w+')
    ls = []
    count = 0
    lsofls=[]
    for i in d:
        i = i.rstrip()
        j = i.split()
        if s in j:
            #w.write(i+'\n')
            count+= len(j)-3
            if count> batchsize:
                lsofls.append(ls)
                ls = []
                count = len(j)-3
            ls.append(i)
    if len(ls) != 0:
        lsofls.append(ls)
    return lsofls
batchsize = BATCH_SIZE
#ls = getsen('swear')
