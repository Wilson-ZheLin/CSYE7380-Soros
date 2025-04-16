import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import re
import os
from tensorflow.keras.models import load_model

# Define these functions BEFORE model load
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

# === Custom Layers ===
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model,
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)
        self.final_dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = (
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["mask"],
        )
        batch_size = tf.shape(query)[0]

        query = self.split_heads(self.query_dense(query), batch_size)
        key = self.split_heads(self.key_dense(key), batch_size)
        value = self.split_heads(self.value_dense(value), batch_size)

        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_logits = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(dk)

        if mask is not None:
            scaled_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_logits, axis=-1)
        attention = tf.matmul(attention_weights, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.num_heads * self.depth))

        return self.final_dense(concat_attention)


# === Customized Transformer Class ===
class CustomizedTransformer:
    from tensorflow.keras.models import load_model

    # Define these functions BEFORE model load
    def create_padding_mask(x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(x):
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padding_mask = create_padding_mask(x)
        return tf.maximum(look_ahead_mask, padding_mask)
    
    def __init__(self, model_dir="models", max_length=40):
        model_path = os.path.join(model_dir, "model.h5")
        tokenizer_path = os.path.join(model_dir, "tokenizer")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(tokenizer_path + ".subwords"):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}.subwords")
        
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "PositionalEncoding": PositionalEncoding,
                "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
                "create_padding_mask": create_padding_mask,
                "create_look_ahead_mask": create_look_ahead_mask,
            },
            compile=False,
        )
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(tokenizer_path)
        self.START_TOKEN = [self.tokenizer.vocab_size]
        self.END_TOKEN = [self.tokenizer.vocab_size + 1]
        self.MAX_LENGTH = max_length

    def preprocess(self, sentence):
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = re.sub(r"i'm", "i am", sentence)
        sentence = re.sub(r"he's", "he is", sentence)
        sentence = re.sub(r"she's", "she is", sentence)
        sentence = re.sub(r"it's", "it is", sentence)
        sentence = re.sub(r"that's", "that is", sentence)
        sentence = re.sub(r"what's", "that is", sentence)
        sentence = re.sub(r"where's", "where is", sentence)
        sentence = re.sub(r"how's", "how is", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"won't", "will not", sentence)
        sentence = re.sub(r"can't", "cannot", sentence)
        sentence = re.sub(r"n't", " not", sentence)
        sentence = re.sub(r"n'", "ng", sentence)
        sentence = re.sub(r"'bout", "about", sentence)
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        return sentence.strip()

    def evaluate(self, sentence):
        sentence = self.preprocess(sentence)
        sentence = tf.expand_dims(self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0)
        output = tf.expand_dims(self.START_TOKEN, 0)

        for _ in range(self.MAX_LENGTH):
            predictions = self.model(inputs=[sentence, output], training=False)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            if tf.equal(predicted_id, self.END_TOKEN[0]):
                break

            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)

    def predict(self, text):
        prediction = self.evaluate(text)
        return self.tokenizer.decode([i for i in prediction if i < self.tokenizer.vocab_size])


# === CLI Entry Point ===
def chat():
    bot = CustomizedTransformer(model_dir="models")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        text = input("You: ")
        if text.strip().lower() in ("exit", "quit"):
            break
        print("Bot:", bot.predict(text))


if __name__ == "__main__":
    chat()
