import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess(filename):

    data = pd.read_csv(filename)

    mails = []
    labels = []

    # Traversing through dataset
    for i in range(len(data)):

        mails.append(data['text'][i])
        labels.append(data['spam'][i])

    # Training set ~ 70%

    train_data = mails[:int(len(data) * 0.8)]
    train_labels = labels[:int(len(data) * 0.8)]

    # Test set ~ 30%

    test_data = mails[int(len(data) * 0.8):]
    test_labels = labels[int(len(data) * 0.8):]

    train_tokenizer = Tokenizer(oov_token="<OOV>")
    train_tokenizer.fit_on_texts(train_data)
    train_word_index = train_tokenizer.word_index
    train_vocab_size = len(train_word_index)

    # Tokenizing train sentences
    train_sequences = train_tokenizer.texts_to_sequences(train_data)
    train_padded = pad_sequences(
        train_sequences, maxlen=50, padding='post', truncating='post')

    # Tokenizing test sentences
    test_sequences = train_tokenizer.texts_to_sequences(test_data)
    test_padded = pad_sequences(
        test_sequences, maxlen=50, padding='post', truncating='post')

    # Converting to numpy arrays
    train_labels_set = np.array(train_labels)
    test_labels_set = np.array(test_labels)
    train_sequences_set = np.array(train_padded)
    test_sequences_set = np.array(test_padded)

    return train_sequences_set, train_labels_set, test_sequences_set, test_labels_set, train_vocab_size
