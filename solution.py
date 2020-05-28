import os
import re
import numpy as np
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Bidirectional, Dense, LSTM
import matplotlib.pyplot as plt

# Set parameters:
batch_size = 256
embedding_dim = 64
epochs = 6
max_len = 500 # mean = std = 255 on training samples
#max_target_len = 20 , mean = 13 & std = 3 on target samples
max_words = 20000

# File names
current_dir = os.getcwd()
folder_name = "ml-unibuc-2020-1"
data_dir = os.path.join(current_dir, folder_name)
train_samples_file = "train_samples.txt"
train_labels_file = "train_labels.txt"
val_source_samples_file = "validation_source_samples.txt"
val_source_labels_file = "validation_source_labels.txt"
val_target_samples_file = "validation_target_samples.txt"
val_target_labels_file = "validation_target_labels.txt"
test_samples_file = "test_samples.txt"
test_labels_file = "sample_submission.csv"

def load_labels(path):
	labels = []
	with open(path, "r") as f:
		lines = f.readlines()

		labels = [int(line[-2]) - 1 for line in lines]

	return np.asarray(labels)


def load_samples():
	samples = []
	IDs = []
	sizes = []
	names = [train_samples_file, val_source_samples_file, val_target_samples_file, test_samples_file]
	for name in names:
		path = os.path.join(data_dir, name)
		with open(path, 'r', encoding='utf8') as f:
			lines = f.readlines()
			nr = len(lines)
			if len(sizes) > 0:
				nr += sizes[-1]
			sizes.append(nr)
			for line in lines:
				pos = line.index('\t')
				IDs.append(line[:pos])
				line = line[pos + 1:]
				#text = re.split('[ \r\n\t\'\"\\-+*#`~=/;:|,.?!“<>(){}]', line)
				#text = [word for word in text if word and word != '$NE$']
				#text = " ".join(text)
				text = line.replace('$NE$', '')
				samples.append(text)

	print("Tokenizing...")
	tokenizer = Tokenizer(num_words=max_words)
	tokenizer.fit_on_texts(samples)
	sequences = tokenizer.texts_to_sequences(samples)
	data = pad_sequences(sequences, maxlen=max_len)
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))

	x_train = data[:sizes[0]]
	x_source_val = data[sizes[0] : sizes[1]]
	x_target_val = data[sizes[1] : sizes[2]]
	x_test = data[sizes[2]:]

	IDs_train = IDs[:sizes[0]]
	IDs_source_val = IDs[sizes[0] : sizes[1]]
	IDs_target_val = IDs[sizes[1] : sizes[2]]
	IDs_test = IDs[sizes[2]:]

	return x_train, x_source_val, x_target_val, x_test, IDs_train, IDs_source_val, IDs_target_val, IDs_test


def write_result(pred, IDs):
	path = os.path.join(data_dir, test_labels_file)
	with open(path, 'w') as f:
		f.write('id,labels\n')
		length = len(IDs)
		for i in range(0, length):
			f.write('{0},{1}\n'.format(IDs[i], pred[i][0] + 1))

# Read the data
print('Reading data...')

# Read samples and IDs
x_train, x_source_val, x_target_val, x_test, IDs_train, IDs_source_val, IDs_target_val, IDs_test = load_samples()

# Read labels
y_train = load_labels(os.path.join(data_dir, train_labels_file))
y_source_val = load_labels(os.path.join(data_dir, val_source_labels_file))
y_target_val = load_labels(os.path.join(data_dir, val_target_labels_file))

# Number of samples
nr_train = len(x_train)
nr_source_val = len(x_source_val)
nr_target_val = len(x_target_val)
nr_test = len(x_test)


# Build the model
print('Building model...')

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(x_source_val, y_source_val))


pred = np.round(model.predict(x_test)).tolist()

write_result(pred, IDs_test)

# Uncomment this to evaluate the validation target data
print(model.evaluate(x_target_val, y_target_val))

# Uncomment this (and the import) to see the loss and accuracy plots

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
