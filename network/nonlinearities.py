import theano.tensor.nnet

def softmax(x):
    return theano.tensor.nnet.softmax(x)

def rectify(x):
    return 0.5 * (x + abs(x))

def identity(x):
    return x
