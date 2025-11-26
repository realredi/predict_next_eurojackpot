import numpy as np
import pandas as pd

from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, LSTM, Dense, Masking
from keras._tf_keras.keras.optimizers import Adam

"""
Simple neural net to predict the next lotto numbers :)
We expect the predicted sequences to resemble the average of all sequences drawn, since 
lotto is close to ideally random, there are no particular patterns in the data. So model should basically default to
the simplest linear approximation: the arithemtic average at each poistion in the sequence.
And indeed the model produces the expected result.

Lotto numbers are categorical not continuous, so we compile model with paramater: categorical_crossentropy
and to represent date we use one-hot arrays:
we represent categorical data (like lottery numbers) numerically so that the model can process it. 
Instead of using the raw number, we create a vector where all elements are 0 except one, which is 1, 
indicating the category.
"""


def to_categorical(y, num_classes):
    """
    y: array of integers, shape (n_samples,)
    num_classes: maximum class index + 1
    shape (n_samples, num_classes)
    returns "one-hot" array:
    """
    y = np.array(y, dtype=int)
    one_hot = np.zeros((y.size, num_classes), dtype=int)
    one_hot[np.arange(y.size), y] = 1
    return one_hot


# load csv, adjust seperator as needed.
df = pd.read_csv(
    r"eurojackpot_allseqs.csv",
    sep=',',
    quoting=1,
    skip_blank_lines=True
)

# fetch valid data
allseq = {}
for index, row in df.iterrows():
    allseq[max(allseq.keys()) + 1 if allseq.keys() else 0] = []
    for value in row:
        if isinstance(value, float) and not np.isnan(value):  # most data looks like integer but is actually float
            allseq[max(allseq.keys())].append(int(value))

# filter sequences with exactly 7 numbers
sequences = [seq for seq in allseq.values() if len(seq) == 7]
sequences = np.array(sequences)

print("final sequences shape:", sequences.shape)
if sequences.shape[1] != 7:
    raise RuntimeError("each sequence must have exactly 7 integers")

num_classes = 50  # Eurojackpot numbers 1–50


# prepare slinding window data
window_size = 5
X, y_raw = [], []

for i in range(len(sequences) - window_size):
    X.append(sequences[i:i+window_size])
    y_raw.append(sequences[i+window_size])  # next sequence

X = np.array(X)  # shape: (samples, window_size, 7)
y_raw = np.array(y_raw)  # shape: (samples, 7)
print("X shape:", X.shape)
print("y_raw shape:", y_raw.shape)

# One-hot encode targets for each number (0-indexed)
y_onehot = np.array([to_categorical(seq-1, num_classes=num_classes) for seq in y_raw])
# shape: (samples, 7, 50)

# build model
input_layer = Input(shape=(window_size, 7))
x = Masking(mask_value=0)(input_layer)
x = LSTM(100, activation='relu', return_sequences=False)(x)

# 7 independent Dense outputs (one per lottery number)
outputs = [Dense(num_classes, activation='softmax', name=f'num_{i+1}')(x) for i in range(7)]

model = Model(inputs=input_layer, outputs=outputs)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']*7  # *7 to match metrics dimensions to output dimensions
)

model.summary()

# training
y_list = [y_onehot[:, i, :] for i in range(7)]  # Keras requires list of outputs
model.fit(X, y_list, epochs=100, verbose=1)


# predict next sequence
def predict_next_sequence(model, last_window):
    last_window = np.array([last_window])  # shape: (1, window_size, 7)
    preds = model.predict(last_window)

    predicted_numbers = []
    for p in preds:
        number = np.argmax(p[0]) + 1  # shift back to 1–50
        predicted_numbers.append(number)
    return predicted_numbers


last_window = sequences[-window_size:]
predicted_sequence = predict_next_sequence(model, last_window)
print("\nPREDICTED NEXT SEQUENCE:", predicted_sequence)
