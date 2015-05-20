def getsen (s = 'flurried'):
    d = open('allsen')
    w = open(s,'w+')
    ls = []
    for i in d:
        j = i.split()
        if s in j:
            w.write(i)
            ls.append(j)
getsen('fortune')
