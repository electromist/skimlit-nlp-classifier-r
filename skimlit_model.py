"""
SkimLit: Sequential Sentence Classification for Medical Abstracts (PubMed 200k RCT).
Replication of paper: "PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts"
Dernoncourt & Lee (2017) - https://arxiv.org/abs/1710.06071
"""

import tensorflow as tf
from tensorflow.keras import layers


def build_tribrid_skimlit_model(
    max_token_len: int = 55,
    max_char_len: int = 290,
    max_vocab_tokens: int = 68000,
    num_classes: int = 5
) -> tf.keras.Model:
    """
    Constructs a multi-modal / tribrid model architecture combining:
    1. Pretrained / Token Embeddings
    2. Character-level Conv1D Embeddings
    3. Positional Line Number Encodings
    4. Positional Total Lines Encodings
    """

    # 1. Token Input & Vectorization / Embedding
    token_inputs = layers.Input(shape=(1,), dtype=tf.string, name="token_input")
    # Placeholder for TextVectorization layer + Embedding layer (or USE)
    token_embed = layers.Dense(128, activation="relu", name="token_dense_projection")

    # 2. Character Input
    char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input")
    char_vectorizer = layers.TextVectorization(
        max_tokens=28,
        output_sequence_length=max_char_len,
        pad_to_max_tokens=True,
        name="char_vectorizer"
    )
    char_embed = layers.Embedding(
        input_dim=28,
        output_dim=25,
        mask_zero=False,
        name="char_embed"
    )
    char_vectors = char_vectorizer(char_inputs)
    char_embeddings = char_embed(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(32), name="char_bi_lstm")(char_embeddings)

    # 3. Line Number Input (One-Hot Encoded)
    line_number_inputs = layers.Input(shape=(15,), dtype=tf.float32, name="line_number_input")
    line_number_dense = layers.Dense(32, activation="relu", name="line_number_dense")(line_number_inputs)

    # 4. Total Lines Input (One-Hot Encoded)
    total_lines_inputs = layers.Input(shape=(20,), dtype=tf.float32, name="total_lines_input")
    total_lines_dense = layers.Dense(32, activation="relu", name="total_lines_dense")(total_lines_inputs)

    # Combine Token & Character embeddings
    token_dense_features = layers.Dense(128, activation="relu")(token_inputs) # Projected feature dimension
    token_char_hybrid = layers.Concatenate(name="token_char_hybrid")([token_dense_features, char_bi_lstm])
    token_char_dense = layers.Dense(256, activation="relu")(token_char_hybrid)
    token_char_dropout = layers.Dropout(0.5)(token_char_dense)

    # Combine Hybrid (Token + Char) with Positional Encodings
    tribrid_embeddings = layers.Concatenate(name="tribrid_concatenate")([
        line_number_dense,
        total_lines_dense,
        token_char_dropout
    ])

    # Final Classification Output
    output_layer = layers.Dense(num_classes, activation="softmax", name="output_layer")(tribrid_embeddings)

    model = tf.keras.Model(
        inputs=[line_number_inputs, total_lines_inputs, token_inputs, char_inputs],
        outputs=output_layer,
        name="SkimLit_Tribrid_Model"
    )

    return model


if __name__ == "__main__":
    print("Building SkimLit model architecture graph...")
    # Summary representation
    print("Model architecture ready for training on PubMed 200k RCT dataset.")
