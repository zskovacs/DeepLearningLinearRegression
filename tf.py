import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# -1 és 1 közé essen
def normalize(array):
    return (array - array.mean()) / array.std()


# az összes adat 70%-a lesz a training sample, a maradék meg a testing
num_train_samples = math.floor(num_house * 0.7)

# array[start:stop:step] -> array[start:] array[:stop] array[::step]
train_house_size = np.asarray(house_size[:num_train_samples])
train_house_price = np.asarray(house_price[:num_train_samples])
train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])
test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

#TENSORFLOW


# y = mx + b
# x: tf_house_size (ez az input)
tf_house_size = tf.placeholder(tf.float32, name="house_size")
# y_: tf_house_price (ez is input, de ezt arra használjuk, hogy ellenőrizzük hogy jól szamoltunk-e)
tf_house_price = tf.placeholder(tf.float32, name="house_price")
# W: tf_size_factor (ez változó -> ehhez kell megtalálnunk a jó értéket)
tf_size_factor = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="Weight")
# b: tf_price_offset (ez változó -> ehhez kell megtalálnunk a jó értéket)
tf_price_offset = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="bias")
# y: tf_price_prediction (azaz enni lesz az ár az adott SIZE-nál)
tf_price_prediction = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# ezzel számoljuk ki, hogy mennyire tárünk el a várt értéktől (2/N E(y - y_)^2)
tf_cost = tf.reduce_mean(tf.pow(tf_price_prediction-tf_house_price,2))/(2*num_train_samples)

# ilyen ütemben fogunk tanulni
learning_rate = 0.1

# az optimizer kiszámolja minden lépésnél a gradienst és aszerint minimalizálja a costot
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    display_every = 200
    num_training_iteration = 5000

    # 50 körben tanítjuk a rendszert
    for iteration in range(num_training_iteration):
        # futtatjuk az optimizert és a cost számolást ahányszor kell (ez utóbbit csak azért, hogy meglegyen az eredmény kiíratáshoz)
        o, c = sess.run([optimizer, tf_cost], feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})

        #kiíratjuk az eredményeket minden második lépésnél
        if(iteration +1 ) % display_every == 0:
            print("iteration #:", '%04d' % (iteration +1), "error=", "{:.9f}".format(c), "m=", sess.run(tf_size_factor), "b=", sess.run(tf_price_offset))

    print("Optimization finished!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size:train_house_size_norm, tf_house_price: train_house_price_norm})
    print("error=", training_cost, "m=", sess.run(tf_size_factor), "b=", sess.run(tf_price_offset))


    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_house_price.mean()
    train_price_std = train_house_price.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_house_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
             label='Learned Regression')
 
    plt.legend(loc='upper left')
    plt.show()