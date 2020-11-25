from sklearn.datasets import make_classification
import tensorflow as tf

X1, y1 = make_classification(n_samples=4000, n_features=6, n_redundant=0,
                             n_clusters_per_class=1, n_classes=3)
learning_rate = 0.01
training_epochs = 600
batch_size = 100

x = tf.placeholder(tf.float32, [None, 6], name='input')  # 6 features
y = tf.placeholder(tf.float32, [None, 3])  # 3 classes

W = tf.Variable(tf.zeros([6, 3]))
b = tf.Variable(tf.zeros([3]))

# softmax回归
pred = tf.nn.softmax(tf.matmul(x, W) + b, name="softmax")
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

prediction_labels = tf.argmax(pred, axis=1, name="output")

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
y2 = tf.one_hot(y1, 3)
y2 = sess.run(y2)

for epoch in range(training_epochs):
    _, c = sess.run([optimizer, cost], feed_dict={x: X1, y: y2})
    if (epoch + 1) % 10 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

print("优化完毕!")
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y2, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = sess.run(accuracy, feed_dict={x: X1, y: y2})
print(acc)
graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
tf.train.write_graph(graph, '.', 'rf.ckpt', as_text=False)
