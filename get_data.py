import random
import numpy as np
import tensorflow as tf
from config import vocab_size, maxlen, batch_size

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import re
import string
import random


# reading data from local dir
shakespear_data = pd.read_csv("./data/Shakespeare_data.csv")
# taking only needed information
PlayerLines = shakespear_data["PlayerLine"]
print(PlayerLines.shape)
PlayerLines= list(PlayerLines)


def preprocess(sequence):
    x = sequence[:, :-1]
    y = sequence[:, 1:]
    # Only print once
    print('sequence',sequence, 'x',x,'y', y)
    return x, y



# num_words - 	the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size - 1)
tokenizer.fit_on_texts(PlayerLines)
sequences = tokenizer.texts_to_sequences(PlayerLines)
# padding	String, "pre" or "post" (optional, defaults to "pre"): pad either before or after each sequence.
sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen + 1, padding="post" )
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html



data = tf.data.Dataset.from_tensor_slices((sequences)).shuffle(256, seed=1).batch(batch_size).map(preprocess)

