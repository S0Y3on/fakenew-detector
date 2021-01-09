def make_models():
    data_dir = './data/'

    import os

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

    print('texts 0:', texts[19])
    print('texts len: ', len(texts))
    print('labels 0: ', labels[0])
    print('labels len: ', len(labels))

    print(texts)
    print(texts[0])

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np
    import math

    validation_ratio = math.floor(len(texts) * 0.3)

    max_words = 10000
    maxlen = 200

    tokenizer = Tokenizer(num_words=max_words)

    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index

    print('전체에서 %s개의 고유한 토큰을 찾았습니다. ' % len(word_index))
    print('word_index type: ', type(word_index))
    print('word_index: ', word_index)

    data = tokenizer.texts_to_sequences(texts)

    print('data 0: ', data[0])
    print('texts 0: ', texts[0])

    data = pad_sequences(data, maxlen=maxlen)

    print('data:', data)
    print('data 0:', data[0])
    print(len(data[0]))
    print(word_index)

    print(type(texts))
    print(type(data))
    print(data.shape)

    def to_one_hot(sequences, dimension):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    data = to_one_hot(data, dimension=max_words)

    labels = np.asarray(labels).astype('float32')

    print(labels)

    print('data: ', data)
    print(len(data[0]))
    print('data[0][0:100]', data[0][0:100])
    print(word_index)

    print('데이터 텐서의 크기: ', data.shape)
    print('레이블 텐서의 크기: ', labels.shape)

    indices = np.arange(data.shape[0])

    np.random.shuffle(indices)

    data = data[indices]

    labels = labels[indices]

    print(indices)

    x_train = data[validation_ratio:]
    y_train = labels[validation_ratio:]

    x_val = data[:validation_ratio]
    y_val = labels[:validation_ratio]

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

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

    print('Accuracy of each epoch:', acc)

    epochs = range(1, len(acc) + 1)

    # import matplotlib.pyplot as plt
    #
    # plt.plot(epochs, acc, 'bo', label='Training Acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation Acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    #
    # plt.figure()
    #
    # plt.plot(epochs, loss, 'bo', label='Training Loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    # plt.title('Training and validation loss')
    # plt.legend()

    # import os
    # from tensorflow.keras.models import load_model
    #
    # loaded_model = load_model('text_binary_model.h5')
    #
    # print("model loaded:", loaded_model)
    #
    # with open('text_binary_tokenizer.pickle', 'rb') as handle:
    #     loaded_tokenizer = pickle.load(handle)
    #
    # test_dir = os.path.join(data_dir, 'test')
    # labels = []
    # texts = []
    #
    # for label_type in ['false', 'true']:
    #     dir_name = os.path.join(test_dir, label_type)
    #     for fname in os.listdir(dir_name):
    #         if fname[-4:] == '.txt':
    #             f = open(os.path.join(dir_name, fname), encoding='utf8')
    #             texts.append(f.read())
    #             f.close()
    #             if label_type == 'false':
    #                 labels.append(0)
    #             else:
    #                 labels.append(1)
    #
    # print('texts:', texts[0])
    # print('texts len:', len(texts))
    #
    # data = loaded_tokenizer.texts_to_sequences(texts)
    #
    # data = pad_sequences(data, maxlen=maxlen)
    #
    # x_test = to_one_hot(data, dimension=max_words)
    #
    # y_test = np.asarray(labels)
    #
    # test_eval = loaded_model.evaluate(x_test, y_test)
    # print('prediction model loss & acc: ', test_eval)
