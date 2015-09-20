import theano
import theano.tensor as T

def sgd( cost, params, lr ):
     grads = T.grad(cost, params)
     updates = []
     for param, grad in zip(params, grads):
         updates.append((param, param - lr*grad))

     return updates

