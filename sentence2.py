def getsen (s,size=5):
    d = open(s)
    ls = []
    count = 0
    lsofls=[]
    for i in d:
        i = i.rstrip()
        j = i.split()
        if True:
            count+= len(j)-3
            if count> batchsize:
                lsofls.append(ls)
                ls = []
                count = len(j)-3
            ls.append(i)
    if len(ls) != 0:
        lsofls.append(ls)
    return lsofls
batchsize = 100
j = getsen('alltest')
