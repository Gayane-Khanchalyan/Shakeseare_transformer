from get_data import data
from contextlib import redirect_stdout
import pickle
import numpy as np
import tensorflow as tf
from config import outputs_path, maxlen,  vocab_size,  embed_dim,  num_heads, feed_forward_dim, num_transformer_blocks
from tensorflow import keras
from tensorflow.keras import layers
from config import vocab_size,maxlen
import time
from get_data import tokenizer
start = time.time()

# from training import model

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu" ), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

@tf.keras.utils.register_keras_serializable()
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim


    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions




def get_transformer_model(
    maxlen,
    vocab_size,
    embed_dim,
    num_heads,
    feed_forward_dim,
    num_transformer_blocks=1
):
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    for i in range(num_transformer_blocks):
        transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
        x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam",
        loss=[loss_fn, None],
        metrics=["accuracy", None]
    )  # No loss and optimization based on word embeddings from transformer block
    return model


class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        tokenizer: Tokenizer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
            self, max_tokens, start_sentence, tokenizer, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.print_every = print_every
        self.k = top_k
        self.start_tokens = tokenizer.texts_to_sequences([start_sentence.split()])[0]

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def generate_text_from_texts(self, texts):
        tokens = tokenizer.texts_to_sequences([texts.split()])[0]
        return self.generate_text_from_tokens(tokens)

    def generate_text_from_tokens(self, start_tokens):
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.max_tokens - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.max_tokens]
                sample_index = self.max_tokens - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)

        txt = self.tokenizer.sequences_to_texts([start_tokens])

        return txt

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return
        txt = self.generate_text_from_tokens(self.start_tokens)
        print(f"generated text:\n{txt}\n")


text_generator = TextGenerator(maxlen, "To be", tokenizer)


# generate the model
model = get_transformer_model(
    maxlen,
    vocab_size,
    embed_dim,
    num_heads,
    feed_forward_dim,
    num_transformer_blocks
)


if __name__ == '__main__':

    keras.backend.clear_session()


    # plot the structure of the model
    keras.utils.plot_model(model, to_file='./output/images/model_structure.png', show_shapes=True)


    # class History_trained_model(object):
    #     def __init__(self, history, epoch, params):
    #         self.history = history
    #         self.epoch = epoch
    #         self.params = params

    # train the model
    model.fit(data, verbose=1, epochs=3 , callbacks=[text_generator])

    # Get the dictionary containing each metric and the loss for each epoch
    history = model.history
    # save the results for further usage
    with open(outputs_path + '/history', 'wb') as file:
        pickle.dump(history, file, pickle.HIGHEST_PROTOCOL)

    # save the summary of the model
    with open(outputs_path + '/textgen_model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # saving model as keras extention
    model.save('./output/model.keras')

stop = time.time()
print(f"Training time: {stop - start}s")