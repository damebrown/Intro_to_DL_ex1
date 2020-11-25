from numpy import loadtxt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
from tensorflow.keras.backend import one_hot
from tensorflow.keras.utils import to_categorical
import matplotlib as mpl
from tensorflow import keras
from tensorflow.python import keras
from itertools import product
from tensorflow.python.keras.utils import losses_utils
import random

mpl.rcParams['figure.figsize'] = (12, 10)

mpl.rcParams['figure.figsize'] = (12, 10)

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

BATCH_SIZE = 2048
EPOCHS = 300
PATIENCE = 5


def make_generator(x, y, batch_size, categorical=True, seed=None):
    num_samples = y.shape[0]
    num_classes = y.shape[1]
    batch_x_shape = (batch_size, *x.shape[1:])
    batch_y_shape = (batch_size, num_classes) if categorical else (batch_size,)
    indexes = [0 for _ in range(num_classes)]
    samples = [[] for _ in range(num_classes)]

    for i in range(num_samples):
        samples[np.argmax(y[i])].append(x[i])

    rand = random.Random(seed)
    while True:
        batch_x = np.ndarray(shape=batch_x_shape, dtype=x.dtype)
        batch_y = np.zeros(shape=batch_y_shape, dtype=y.dtype)
        for i in range(batch_size):
            random_class = rand.randrange(num_classes)
            current_index = indexes[random_class]
            indexes[random_class] = (current_index + 1) % len(samples[random_class])
            if current_index == 0:
                rand.shuffle(samples[random_class])
            batch_x[i] = samples[random_class][current_index]
            if categorical:
                batch_y[i][random_class] = 1
            else:
                batch_y[i] = random_class
        batch_y = np.expand_dims(batch_y, axis=1)

        yield batch_x, batch_y


class MyModel(Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = Flatten(input_shape=(9, 20))
        self.fc1 = Dense(9, activation='relu')
        self.drop1 = Dropout(0.5)
        self.fc2 = Dense(9, activation='relu')
        self.drop2 = Dropout(0.5)
        self.fc3 = Dense(9, activation='relu')
        self.drop3 = Dropout(0.5)
        self.fc4 = Dense(15, activation='relu')
        self.drop4 = Dropout(0.5)

        self.out = Dense(1, activation='sigmoid')

    def call(self, x, **kwargs):
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.drop1(x)
        # x = self.fc2(x)
        # x = self.drop2(x)
        # x = self.fc3(x)
        # x = self.drop3(x)
        # x = self.fc4(x)
        # x = self.drop4(x)

        return self.out(x)

    def get_config(self):
        pass


def weighted_binary_crossentropy(y_true, y_pred, weights=None):
    if weights is None:
        weights = [1., 8.]
    assert len(weights) == 2
    tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
    tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)
    weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
    ce = K.binary_crossentropy(tf_y_true, tf_y_pred)
    loss = K.mean(tf.multiply(ce, weights_v))
    return loss


@tf.function
def train_step(X, y, model, metrics, loss_ob, loss_train, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(X, training=True)
        loss = loss_ob(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_train(loss)
    update_metrics(metrics, predictions, y)


@tf.function
def val_step(x, y, model, metrics, loss_ob, loss_val):
    predictions = model(x, training=False)
    t_loss = loss_ob(y, predictions)
    loss_val(t_loss)
    update_metrics(metrics, predictions, y)


def update_metrics(metrics, predictions, y):
    for met in metrics.values():
        met(y, predictions)


def encoder(data):
    input_features = []
    for sequence in data:
        integer_encoded = np.array([ord(c.upper()) - ord('A') for c in sequence])
        x = one_hot(integer_encoded, 20)
        input_features.append(x)
    return np.stack(input_features)


def get_data(neg, pos):
    X = np.concatenate((neg, pos))
    y = np.array([0] * neg.shape[0] + [1] * pos.shape[0], dtype='float32')
    return X, y


def train(model, train, val, loss, optimizer, test=None, epochs=50):
    train_loss, train_metrics = get_loss_acc('train')
    val_loss, metrics = get_loss_acc('test')
    l_train = []
    l_test = []
    num_batches = 5
    for epoch in range(epochs):
        i = 0
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        val_loss.reset_states()

        for met in train_metrics.values():
            met.reset_states()

        for met in metrics.values():
            met.reset_states()
        for x, y in train:
            i += 1
            if i > num_batches:
                break
            train_step(x, y, model, metrics, loss, train_loss, optimizer)

        # for x, y in train:
        #     train_step(x, y, model, metrics, loss, train_loss, optimizer)

        for val_x, val_y in val:
            val_step(val_x, val_y, model, metrics, loss, val_loss)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Loss Val: {val_loss.result()}, '
            f'auc: {metrics["auc"].result()}, '
            f'recall: {metrics["recall"].result()}, '
            f'precision: {metrics["precision"].result()}, '
        )
        l_test.append(val_loss.result())
        l_train.append(train_loss.result())

    plot_loss(l_test, l_train)


def plot_loss(l_test, l_train):
    plt.title('Loss : Train vs Validation')
    plt.plot(l_test, label='Test')
    plt.plot(l_train, label='Train')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Crossentropy')
    plt.legend()
    plt.show()


def get_loss_acc(data_name):
    loss = tf.keras.metrics.Mean(name=data_name + '_loss')
    met = {'tp': keras.metrics.TruePositives(name='tp'),
           'fp': keras.metrics.FalsePositives(name='fp'),
           'tn': keras.metrics.TrueNegatives(name='tn'),
           'fn': keras.metrics.FalseNegatives(name='fn'),
           'accuracy': keras.metrics.BinaryAccuracy(name='accuracy'),
           'precision': keras.metrics.Precision(name='precision'),
           'recall': keras.metrics.Recall(name='recall'),
           'auc': keras.metrics.AUC(name='auc'),
           }
    return loss, met


def plot_cm(labels, predictions, p=0.5, model_name='', print_results=False):
    cm = confusion_matrix(labels, predictions > p, normalize='true')
    sns.heatmap(cm, annot=True, fmt=".2f")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if model_name:
        plt.title(model_name)
    if print_results:
        print('True Negatives: ', cm[0][0])
        print('False Positives: ', cm[0][1])
        print('False Negatives: ', cm[1][0])
        print('True Positives: ', cm[1][1])
        print('Total Fraudulent Transactions: ', np.sum(cm[1]))
    plt.show()


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)
    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.grid(True)
    # plt.xlim([-0.5, 30])
    # plt.ylim([80, 100.5])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend(loc='lower right')


def compare_cm(X_test, models, y_test):
    for i, model in enumerate(models):
        plt.subplot(2, 2, i + 1)
        test_predictions = model(X_test, training=False)
        y_test = np.argmax(y_test, axis=1)
        test_predictions = np.argmax(test_predictions, axis=1)
        plot_cm(y_test, test_predictions, model_name='Model ' + str(i))


def main():
    neg = loadtxt('neg_A0201.txt', dtype=str)
    pos = loadtxt('pos_A0201.txt', dtype=str)

    X, y = get_data(neg, pos)
    y = np.expand_dims(y, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.9, random_state=42)
    X_train, X_val, X_test = encoder(X_train), encoder(X_val), encoder(X_test)

    optimizer = tf.keras.optimizers.Adam()
    # loss_object = weighted_binary_crossentropy
    loss_object = tf.keras.losses.BinaryCrossentropy()
    train_ds = make_generator(X_train, to_categorical(y_train), batch_size=BATCH_SIZE, categorical=False)
    # train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    model = MyModel()
    train(model, train=train_ds, val=val_ds, loss=loss_object, optimizer=optimizer, epochs=150)

    test_predictions = model(X_test, training=False)
    train_predictions = model(X_train, training=False)
    plot_cm(y_test, test_predictions, model_name='Model')

    plot_roc("Test", y_test, test_predictions, linestyle='--')
    plot_roc("Train", y_train, train_predictions)
    plt.show()


if __name__ == '__main__':
    main()
