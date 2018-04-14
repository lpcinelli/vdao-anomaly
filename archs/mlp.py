from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten
from keras.regularizers import l1, l2


def mlp(input_shape, neurons_per_layer, weight_decay=0):
    """ Multilayer Perceptron (Fully-Connected) network
    """
    classifier = Sequential()
    classifier.add(Flatten(input_shape=input_shape))

    for idx, nb_neurons in enumerate(neurons_per_layer):
        classifier.add(Dense(nb_neurons,
                             name='Dense_feat_{}'.format(idx),
                             kernel_regularizer=l2(weight_decay)))
        classifier.add(Activation('relu'))

    classifier.add(Dense(1, name='Dense_feat',
                         kernel_regularizer=l2(weight_decay)))
    classifier.add(Activation('sigmoid'))

    return classifier
