from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.datasets import load_iris

from imblearn.keras import BalancedBatchGenerator


iris = load_iris()
X, y = iris.data, to_categorical(iris.target, 3)


def test_balanced_batch_generator():
    model = Sequential()
    model.add(Dense(y.shape[1], input_dim=X.shape[1], activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    training_generator = BalancedBatchGenerator(X, y)
    model.fit_generator(generator=training_generator,
                        epochs=10,
                        verbose=10)
