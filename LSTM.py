from dataprep import preprocess
import tensorflow as tf
import matplotlib.pyplot as plt

# Split data into train and validation tests

train_data, train_labels, test_data, test_labels, train_vocab_size = preprocess(
    r'C:\Users\praat\OneDrive\Desktop\Sem VI\DWDM_Project\emaildata.csv')

# LSTM Model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(train_vocab_size+1, 100, input_length=50),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Providing loss function and optimizer

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=10,
                    validation_data=(test_data, test_labels), verbose=2)

# Plotting graphs to monitor loss and accuracy metrics


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, "loss")
