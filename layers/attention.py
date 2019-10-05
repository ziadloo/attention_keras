import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
import functools


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, attention_dim=None, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        assert isinstance(input_shape, dict)
        assert 'values' in input_shape
        assert 'query' in input_shape
        
        # Create a trainable weight variable for this layer.
        if not isinstance(input_shape["values"], list):
            input_shape["values"] = [input_shape["values"]]
        if self.attention_dim is None:
            self.attention_dim = int(input_shape["values"][0][-1])

        self.W_a = self.add_weight(name='W_a',
                                   shape=(sum([int(s[-1]) for s in input_shape["values"]]), self.attention_dim), # (lat1+lat2+...+latn, d3)
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=(int(input_shape["query"][-1]), self.attention_dim), # (d2, d3)
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=(self.attention_dim, 1), # (d3, 1)
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build([input_shape["values"][0], input_shape["query"]])  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert isinstance(inputs, dict)
        assert 'values' in inputs
        assert 'query' in inputs

        if not isinstance(inputs["values"], list):
            inputs["values"] = [inputs["values"]]
        
        encoder_out_seq, decoder_out_seq = inputs["values"], inputs["query"]
        if verbose:
            print('encoder_out_seq>', [eos.shape for eos in encoder_out_seq])
            print('decoder_out_seq>', decoder_out_seq.shape)

        """ Computing h.Wa where h=[h0, h1, ..., hi]"""
        # <= (batch_size, en_seq_len, latent_dim|d3)
        hidden_size_product = functools.reduce(lambda a, b: a * b,
                                               [K.int_shape(eos)[-2] for eos in encoder_out_seq], 1)
        latent_size_sum = sum([K.int_shape(eos)[-1] for eos in encoder_out_seq])
        repeated_encoder_out = [K.repeat_elements(eos, int(hidden_size_product / K.int_shape(eos)[-2]), axis=-2)
                                for eos in encoder_out_seq]
        # (batch_size, h1*h2*...*hn, lat1+lat2+...+latn)
        hiddens_combined = K.concatenate(repeated_encoder_out)
        # (batch_size*h1*h2*...*hn, lat1+lat2+...+latn)
        hiddens_combined = K.reshape(hiddens_combined, (-1, latent_size_sum))
        # (batch_size*h1*h2*...*hn, d3)
        W_a_dot_h = K.dot(hiddens_combined, self.W_a)
        # (batch_size, h1*h2*...*hn, d3)
        W_a_dot_h = K.reshape(W_a_dot_h, (-1, hidden_size_product, self.attention_dim))
        # (batch_size, h1*h2*...*hn, lat1+lat2+...+latn) for later
        hiddens_combined = K.reshape(hiddens_combined, (-1, hidden_size_product, latent_size_sum))

        if verbose:
            print('wa.s>', W_a_dot_h.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            # input: (batch_size, latent_dim)
            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Computing sj.Ua """
            # (batch_size, 1, d3)
            U_a_dot_s = K.expand_dims(K.dot(inputs, self.U_a), 1)
            if verbose:
                print('Ua.h>', U_a_dot_s.shape)

            """ tanh(h.Wa + s.Ua) """
            # (batch_size, h1*h2*...*hn, d3) = (batch_size, h1*h2*...*hn, d3) + (batch_size, 1, d3)
            Wh_plus_Us = K.tanh(W_a_dot_h + U_a_dot_s)
            # (batch_size*h1*h2*...*hn, d3)
            reshaped_Wh_plus_Us = K.reshape(Wh_plus_Us, (-1, self.attention_dim))
            if verbose:
                print('Wh+Us>', reshaped_Wh_plus_Us.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # (batch_size*h1*h2*...*hn, 1)
            Wh_plus_Us_dot_Va = K.dot(reshaped_Wh_plus_Us, self.V_a)
            # <= (batch_size, h1*h2*...*hn)
            e_i = K.reshape(Wh_plus_Us_dot_Va, (-1, hidden_size_product))
            e_i = K.softmax(e_i)
            #e_i = K.reshape(e_i, [-1] + [K.int_shape(eos)[-2] for eos in encoder_out_seq])

            if verbose:
                print('ei>', e_i.shape)

            # (batch_size, h1*h2*...*hn)
            return e_i, states

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # (batch_size, lat1+lat2+...+latn) = (batch_size, h1*h2*...*hn, lat1+lat2+...+latn) * (batch_size, h1*h2*...*hn, 1)
            c_i = K.sum(hiddens_combined * K.expand_dims(inputs, -1), axis=-2)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, states

        # <= (batch_size, enc_seq_len, latent_dim   encoder_out_seq.shape[-2]
        fake_state_e = K.zeros_like(K.placeholder(shape=(1, 1))) #decoder_out_seq.shape[0]
        fake_state_c = K.zeros_like(K.placeholder(shape=(1, 1)))

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        hidden_size_product = functools.reduce(lambda a, b: a * b,
                                               [s[-2] for s in input_shape["values"]], 1)
        latent_size_sum = sum([s[-1] for s in input_shape["values"]])

        return [
            tf.TensorShape((input_shape["query"][0], input_shape["query"][-2], latent_size_sum)),
            tf.TensorShape((input_shape["query"][0], input_shape["query"][-2], hidden_size_product))
        ]
    
    def get_config(self):
        return {'attention_dim': self.attention_dim}
