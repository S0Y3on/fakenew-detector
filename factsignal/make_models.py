
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import math
import os

def make_models():
    data_dir = './data/'
    train_dir = os.path.join(data_dir, 'train')

    labels = []
    texts = []

    for label_type in ['false', 'true']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname), encoding='utf8')
                texts.append(f.read())
                f.close()
                if label_type == 'false':
                    labels.append(0)
                else:
                    labels.append(1)

    validation_ratio = math.floor(len(texts) * 0.3)

    max_words = 10000
    maxlen = 200

    tokenizer = Tokenizer(num_words=max_words)

    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index

    data = tokenizer.texts_to_sequences(texts)

    print('data 0: ', data[0])
    print('texts 0: ', texts[0])

    data = pad_sequences(data, maxlen=maxlen)

    def to_one_hot(sequences, dimension):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    data = to_one_hot(data, dimension=max_words)

    labels = np.asarray(labels).astype('float32')

    print(labels)

    indices = np.arange(data.shape[0])

    np.random.shuffle(indices)

    data = data[indices]

    labels = labels[indices]

    print(indices)

    x_train = data[validation_ratio:]
    y_train = labels[validation_ratio:]

    x_val = data[:validation_ratio]
    y_val = labels[:validation_ratio]


    model = Sequential()

    model.add(Dense(64, activation='relu', input_shape=(max_words,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

    history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val))

    history_dict = history.history

    model.save('datatext_binary_model.h5')

    import pickle

    with open('datatext_binary_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
