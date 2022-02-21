import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def layer(output_dim, input_dim, inputs, activation=None):
    # 参数随机初始化
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    if activation == None:
        outputs = tf.matmul(inputs, W) + b
    else:
        outputs = activation(tf.matmul(inputs, W) + b)
    return outputs


def mse(mnist):
    # 输入层
    x = tf.placeholder("float", [None, 784])
    # 隐层 sigmoid激活函数
    h1 = layer(output_dim=256, input_dim=784, inputs=x, activation=tf.nn.sigmoid)
    # 输出层
    y_predict = layer(output_dim=10, input_dim=256, inputs=h1)
    y_label = tf.placeholder("float", [None, 10])
    # 均方误差损失函数 无正则化
    loss_function = tf.reduce_mean(tf.square(y_predict - y_label))
    # 固定学习率
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss_function)
    correct_prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 训练参数
    trainEpochs = 50
    batchSize = 100
    total_batchs_list = range(int(mnist.train.num_examples / batchSize))
    epoch_list = list(range(trainEpochs))
    loss_list = []
    accuracy_list = []
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # 训练
    for epoch in epoch_list:
        for _ in total_batchs_list:
            batch_x, batch_y = mnist.train.next_batch(batchSize)
            sess.run(optimizer, feed_dict={x: batch_x, y_label: batch_y})
        # 使用训练集计算训练过程中的准确率
        loss, acc = sess.run([loss_function, accuracy], feed_dict={x: mnist.train.images, y_label: mnist.train.labels})
        loss_list.append(loss)
        accuracy_list.append(acc)
        print("均方误差训练轮数:", '%02d' % (epoch + 1), "Loss=", "{:.9f}".format(loss), "训练集准确率=", acc)

    # 使用测试集计算模型最终的准确率
    print("均方误差测试集准确率:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))
    return epoch_list, accuracy_list
