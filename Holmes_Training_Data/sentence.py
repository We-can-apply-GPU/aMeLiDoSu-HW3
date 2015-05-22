def getsen (s = 'flurried'):
    d = open('allsen')
    w = open(s,'w+')
    ls = []
    lsofls=[]
    for i in d:
        j = i.split()
        if s in j:
            w.write(i)
            ls.append(i.rstrip())
            if len(ls) == batchsize:
                lsofls.append(ls)
                ls = []
    if len(ls) != 0:
        for i in range( batchsize-len(ls)):
            ls.append(ls[-1])
        lsofls.append(ls)
    return lsofls
batchsize = 10
ls = getsen('fortune')
