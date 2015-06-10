import numpy
import theano
import theano.tensor as T
import cPickle
import gzip

def main(n_inputs=784,
         n_hiddens0=1000,
         n_hiddens1=500,
         n_hiddens2=250,
         n_hiddens3=30,
         learning_rate=5e-4,
         epsilon=0.0001,
         momentum=0.740728016058,
         n_updates=3000 * 250,
         distribution="uniform",
         batch_size=200,
         restart=0,
         state=None,
         channel=None,
         **kwargs):
    numpy.random.seed(0xeffe)

    print locals()
    
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = cPickle.load(gzip.open("/data/lisatmp/dauphiya/ddbm/mnist.pkl.gz", 'rb'))
    
    num_examples = train_x.shape[0]
    
    ### Create shared variables
    
    def init_param(n_x, n_y):
        return theano.shared(numpy.random.normal(size=(n_x, n_y)))
    
    W0 = init_param(n_inputs, n_hiddens0)
    W1 = init_param(n_hiddens0, n_hiddens1)
    W2 = init_param(n_hiddens1, n_hiddens2)
    W3 = init_param(n_hiddens2, n_hiddens3)
    W0_ = init_param(n_hiddens0, n_inputs)
    W1_ = init_param(n_hiddens1, n_hiddens0)
    W2_ = init_param(n_hiddens2, n_hiddens1)
    W3_ = init_param(n_hiddens3, n_hiddens2)
    b0 = theano.shared(numpy.zeros(n_hiddens0, 'float32'))
    b1 = theano.shared(numpy.zeros(n_hiddens1, 'float32'))
    b2 = theano.shared(numpy.zeros(n_hiddens2, 'float32'))
    b3 = theano.shared(numpy.zeros(n_hiddens3, 'float32'))
    b0_ = theano.shared(numpy.zeros(n_inputs, 'float32'))
    b1_ = theano.shared(numpy.zeros(n_hiddens0, 'float32'))
    b2_ = theano.shared(numpy.zeros(n_hiddens1, 'float32'))
    b3_ = theano.shared(numpy.zeros(n_hiddens2, 'float32'))
    params = [W0, b0, W1, b1, W2, b2, W3, b3, W0_, b0_, W1_, b1_, W2_, b2_, W3_, b3_]

    input = T.matrix('x')
    mu = theano.shared(numpy.float32(1.0))
    eta1 = theano.shared(numpy.float32(1e-1))
    eta2 = theano.shared(numpy.float32(1e-1))
    
    z1 = T.matrix('z1')
    z2 = T.matrix('z2')
    z3 = T.matrix('z3')
    z4 = T.matrix('z4')
    z3_ = T.matrix('z3_')
    z2_ = T.matrix('z2_')
    z1_ = T.matrix('z1_')
    
    Z = [z1, z2, z3, z4, z3_, z2_, z1_]
    
    hidden1 =  T.nnet.sigmoid(T.dot(input, W0) + b0)
    hidden2 =  T.nnet.sigmoid(T.dot(hidden1, W1) + b1)
    hidden3 =  T.nnet.sigmoid(T.dot(hidden2, W2) + b2)
    hidden4 =  T.dot(hidden3, W3) + b3
    
    hidden3_ = T.nnet.sigmoid(T.dot(hidden4, W3_) + b3_)
    hidden2_ = T.nnet.sigmoid(T.dot(hidden3_, W2_) + b2_)
    hidden1_ = T.nnet.sigmoid(T.dot(hidden2_, W1_) + b1_)
    output = T.nnet.sigmoid(T.dot(hidden1_, W0_) + b0_)
    
    loss = -(input * T.log(output) + (1. - input) * T.log(1. - output)).sum(1).mean()
    
    h2 = T.nnet.sigmoid(T.dot(z1, W1) + b1)
    h3 = T.nnet.sigmoid(T.dot(z2, W2) + b2)
    h4 = T.nnet.sigmoid(T.dot(z3, W3) + b3)
    h3_ = T.nnet.sigmoid(T.dot(z4, W3_) + b3_)
    h2_ = T.nnet.sigmoid(T.dot(z3_, W2_) + b2_)
    h1_ = T.nnet.sigmoid(T.dot(z2_, W1_) + b1_)
    out = T.nnet.sigmoid(T.dot(z1_, W0_) + b0_)
    
    mac_loss = -(input * T.log(out) + (1. - input) * T.log(1. - out)).sum(1).mean()\
               + mu * ((z1 - hidden1)**2).sum(1).mean()\
               + mu * ((z2 - h2)**2).sum(1).mean()\
               + mu * ((z3 - h3)**2).sum(1).mean()\
               + mu * ((z4 - h4)**2).sum(1).mean()\
               + mu * ((z3_ - h3_)**2).sum(1).mean()\
               + mu * ((z2_ - h2_)**2).sum(1).mean()\
               + mu * ((z1_ - h1_)**2).sum(1).mean()
               
    grad_Z = theano.grad(mac_loss, Z)
    new_Z = [z - eta1*g_z for z, g_z in zip(Z, grad_Z)]
    grad_params = theano.grad(mac_loss, params)
    new_params = [p - eta2*g_p for p, g_p in zip(params, grad_params)]
    
    loss_fun = theano.function([input], loss)
    Z_fun = theano.function([input] + Z, new_Z)
    param_fun = theano.function([input] + Z, mac_loss, updates=zip(params, new_params))
    
    
    Z1 = numpy.random.normal(loc=0.5, scale=0.1, size=(num_examples, n_hiddens0)).astype('float32')
    Z2 = numpy.random.normal(loc=0.5, scale=0.1, size=(num_examples, n_hiddens1)).astype('float32')
    Z3 = numpy.random.normal(loc=0.5, scale=0.1, size=(num_examples, n_hiddens2)).astype('float32')
    Z4 = numpy.random.normal(loc=0.5, scale=0.1, size=(num_examples, n_hiddens3)).astype('float32')
    Z3_ = numpy.random.normal(loc=0.5, scale=0.1, size=(num_examples, n_hiddens2)).astype('float32')
    Z2_ = numpy.random.normal(loc=0.5, scale=0.1, size=(num_examples, n_hiddens1)).astype('float32')
    Z1_ = numpy.random.normal(loc=0.5, scale=0.1, size=(num_examples, n_hiddens0)).astype('float32')
    
    print loss_fun(train_x)
    n_batches = int(numpy.ceil(num_examples/batch_size))
    
    for update in range(n_updates):
        # W-step
        for _ in range(5):
            for index in range(n_batches):
                inputs = [y[index*batch_size:(index+1)*batch_size, :] for y in [train_x, Z1, Z2, Z3, Z4, Z3_, Z2_, Z1_]]
                param_fun(*inputs)
                
        # Z-step
        for index in range(n_batches):
            inputs = [y[index*batch_size:(index+1)*batch_size, :] for y in [train_x, Z1, Z2, Z3, Z4, Z3_, Z2_, Z1_]]
            tmp = Z_fun(*inputs)
            Z1[index*batch_size:(index+1)*batch_size] = tmp[0]
            Z2[index*batch_size:(index+1)*batch_size] = tmp[1]
            Z3[index*batch_size:(index+1)*batch_size] = tmp[2]
            Z4[index*batch_size:(index+1)*batch_size] = tmp[3]
            Z3_[index*batch_size:(index+1)*batch_size] = tmp[4]
            Z2_[index*batch_size:(index+1)*batch_size] = tmp[5]
            Z1_[index*batch_size:(index+1)*batch_size] = tmp[6]
            
        print loss_fun(train_x)    
    
if __name__ == "__main__":
    main()
    