import numpy as np
import tensorflow as tf
from get_data import sequences, tokenizer
from config import  maxlen, vocab_size
from tensorflow import keras
from training import model, text_generator
from get_data import tokenizer


# load the optimal weights
model.load_weights('./output/model.keras')

shakespearin_text = ['To be, or not to be: that is the question.', 'If music be the food of love, play on.']
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size - 1)
tokenizer.fit_on_texts(shakespearin_text)
sequences = tokenizer.texts_to_sequences(shakespearin_text)
sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding="post")

print(text_generator.generate_text_from_tokens(list(sequences[0][0:2])))
