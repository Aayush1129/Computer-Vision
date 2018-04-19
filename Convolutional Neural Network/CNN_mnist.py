import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
num_epoch = 500
batch_size = 16
input_size = 784
num_class = 10

X = tf.placeholder(tf.float32, shape=[None, input_size])
Y = tf.placeholder(tf.float32, shape=[None, num_class])
keep_prob = tf.placeholder(tf.float32)

def conv_2d(X, W, b, stride = 1):
    X = tf.nn.conv2d(X, W, strides = [1, stride, stride, 1], padding = 'SAME')
    X = tf.nn.bias_add(X, b)
    return tf.nn.relu(X)

def max_pool(X, k = 2, stride = 2):
    X = tf.nn.max_pool(X, ksize = [1, k, k ,1], strides = [1, stride, stride, 1], padding = 'SAME')
    return X

def conv_net(X, weights, bias, dropout):
    X = tf.reshape(X, shape = [-1,28,28,1])
    
    X = conv_2d(X, weights['wc1'], bias['b1'])
    X = max_pool(X)

    X = conv_2d(X, weights['wc2'], bias['b2'])
    X = max_pool(X)
    X = tf.reshape(X, shape = [-1, weights['w_fc1'].get_shape().as_list()[0]])

    X = tf.add(tf.matmul(X, weights['w_fc1']),bias['b_fc1'])
    X = tf.nn.relu(X)
    X = tf.nn.dropout(X, dropout)
    X = tf.add(tf.matmul(X, weights['w_out']), bias['b_out'])
    return X
    

weights = {
    'wc1' : tf.Variable(tf.random_normal(shape=[5,5,1,32])),
    'wc2' : tf.Variable(tf.random_normal(shape=[5,5,32,64])),
    'w_fc1' : tf.Variable(tf.random_normal(shape=[7*7*64, 1024])),
    'w_out' : tf.Variable(tf.random_normal(shape=[1024, num_class]))
}


bias = {
    'b1' : tf.Variable(tf.random_normal(shape=[32])),
    'b2' : tf.Variable(tf.random_normal(shape=[64])),
    'b_fc1' : tf.Variable(tf.random_normal(shape=[1024])),
    'b_out' : tf.Variable(tf.random_normal(shape=[num_class]))
}

logits = conv_net(X, weights, bias, keep_prob)
pred = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    num_iteration = int(mnist.train.num_examples/batch_size)
    for epoch in range(num_epoch):
        avg_accur = 0
        for iter in range(num_iteration):
            batch_X,batch_Y = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={X: batch_X, Y: batch_Y, keep_prob: 0.8})
            accur = sess.run(accuracy, feed_dict={X: batch_X, Y: batch_Y, keep_prob: 0.8})
            avg_accur+=accur/num_epoch
        print('Epoch: ', '%.4d'%(epoch+1), 'cost: ', '{:.9f}'.format(c), 'Average Accuracy: ', '{: .9f}'.format(avg_accur))
    test_accur = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
    print('Final Test Accuracy: ', '{:.9f}'.format(test_accur))

