from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
from sklearn import metrics

path='./path/Iris_Imgs'

# set images to 200*200
w = 200
h = 200
c = 3

# read images with folders's names as labels
def read_img(path):
    cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

data, label = read_img(path)


# # disorder images and labels and split train and test dataset(If dataset is large)
# num_example=data.shape[0]
# arr=np.arange(num_example)
# np.random.shuffle(arr)
# data=data[arr]
# label=label[arr]
#
# ratio=0.8
# s=np.int(num_example*ratio)
# x_train=data[:s]
# y_train=label[:s]
# x_val=data[s:]
# y_val=label[s:]

# split dataset (If dataset is small)
x_train=np.empty((120,200,200,3))
y_train=np.empty(120,dtype=int)
x_val=np.empty((30,200,200,3))
y_val=np.empty(30,dtype=int)

for i in range(3):
    x_train[i*40:i*40+40]=data[i*50:i*50+40]
    y_train[i*40:i*40+40]=label[i*50:i*50+40]
    x_val[i*10:i*10+10]=data[i*50+40:i*50+50]
    y_val[i*10:i*10+10]=label[i*50+40:i*50+50]

# define minibatches
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#-----------------build CNN----------------------
# placeholders
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')


def CNNlayer():
    # 1st Conv(200——>100)
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 2nd Conv(100->50)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 3rd Conv(50->25)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # flatten the layer
    flattened = tf.contrib.layers.flatten(pool3)

    # Fully connected
    dense1 = tf.layers.dense(inputs=flattened,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
                             )

    logits = tf.layers.dense(inputs=dense1,
                             units=3,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
                             )
    # softmax layers
    softmax = tf.contrib.layers.softmax(logits)

    return softmax

#-----------------CNN ends----------------------

# setup tensors
softmax = CNNlayer()
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=softmax)
train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(softmax,1),tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predict = tf.argmax(softmax, 1)

# setup the number of model to save
saver = tf.train.Saver(max_to_keep=1)
max_acc = 0
f = open('./path/acc.csv', 'w')

# Setup the epochs and batch size
n_epoch = 40
batch_size = 15


print("Starting Training Session...")
with tf.Session() as sess:
    # initialize the 1st session
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        start_time = time.time()

        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))

        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac, pred = sess.run([loss, acc, predict], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("   validation loss: %f" % (val_loss / n_batch))
        print("   validation acc: %f\n" % (val_acc / n_batch))

        # save the training result in a CSV file
        f.write(str(epoch + 1) + ', val_acc, ' + str(val_acc/ n_batch) + '\n')

        # store the parameters of trained CNN with highest validation accuracy
        if val_acc > max_acc:
            max_acc = val_acc
            saver.save(sess,
                       './path/ckpt/flower.ckpt')
f.close()

print("Starting Predicting Session...")
with tf.Session() as sess:
    # initialize the 2nd session
    sess.run(tf.global_variables_initializer())

    # loade the graph of the model
    saver = tf.train.import_meta_graph(
            './path/ckpt/flower.ckpt.meta')
    # restore the parameters of the model
    saver.restore(sess,
            './path/ckpt/flower.ckpt')
    print("Model restored.\n")

    # show the result of prediction on test data
    pred = sess.run(predict, feed_dict={x: x_val})
    accuracy = metrics.accuracy_score(y_val,pred)
    print('Predicted label:', pred)
    print('  True label   :', y_val)
    print('   Accuracy    :', accuracy)

    # show confusion matrix
    print('\n confusion matrix:\n', metrics.confusion_matrix(y_val, pred))

