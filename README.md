# Keras Attention Layer

## Version (s)

- TensorFlow: 1.14.0 (Tested)
- TensorFlow: 2.0 (Tested)

## Introduction

This is an implementation of Attention (only supports [Bahdanau Attention](https://arxiv.org/pdf/1409.0473.pdf) right now) in Keras. This implementation also supports multiple joint values as the input with a single query.

## Project structure

```
data (sample data for the examples)
 |--- en.model
 |--- en.vocab
 |--- fr.model
 |--- fr.vocab
 └--- lstm_weights.h5
layers
 └--- attention.py (Attention implementation)
examples
 └--- colab
   └--- LSTM.ipynb (Jupyter notebook to be run on Google Colab)
 └--- nmt
   |--- model.py (NMT model defined with Attention)
   └--- train.py ( Code for training/inferring/plotting attention with NMT model)
 └--- nmt_bidirectional
   |--- model.py (NMT birectional model defined with Attention)
   └--- train.py ( Code for training/inferring/plotting attention with NMT model)
results (created by train_nmt.py to store model)

```
## How to use

Just like you would use any other `tensoflow.python.keras.layers` object.

```python
from attention_keras.layers.attention import AttentionLayer

attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer({"values": encoder_outputs, "query": decoder_outputs})

```

Or as for the multiple joint values:

```python
from attention_keras.layers.attention import AttentionLayer

attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer({"values": [encoder1_outputs, encoder2_outputs],
                                    "query": decoder_outputs})

```

Here,

- `encoder_outputs` - Sequence of encoder ouptputs returned by the RNN/LSTM/GRU (i.e. with `return_sequences=True`)
- `decoder_outputs` - The above for the decoder
- `attn_out` - Output context vector sequence for the decoder. This is to be concat with the output of decoder (refer `model/nmt.py` for more details)
- `attn_states` - Energy values if you like to generate the heat map of attention (refer `model.train_nmt.py` for usage)

## Visualizing Attention weights

An example of attention weights can be seen in `model.train_nmt.py`

After the model trained attention result should look like below.

![Attention heatmap](https://github.com/ziadloo/attention_keras/blob/master/results/attention.png)

The same plot but for a model trained with sub-words as tokens.

![Attention heatmap](https://github.com/ziadloo/attention_keras/blob/master/results/attention_scores_subword.png)

## Running the examples

In order to run the example you need to download `small_vocab_en.txt` and `small_vocab_fr.txt` from [Udacity deep learning repository](https://github.com/udacity/deep-learning/tree/master/language-translation/data) and place them in the `data` folder.

Also, there's an LSTM version of the same example in Colab, just follow the instructions [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziadloo/attention_keras/blob/master/examples/colab/LSTM.ipynb)

___

If you have improvements (e.g. other attention mechanisms), contributions are welcome!

## Disclaimer

The original credit goes to [thushv89](https://github.com/thushv89/attention_keras)
