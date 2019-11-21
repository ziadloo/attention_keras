import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
import functools


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_h, U_a, and V_a
     """

    def __init__(self, attention_dim=None, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.W_h = []
        self.U_a = None
        self.V_a = None

    def build(self, input_shape):
        assert isinstance(input_shape, dict)
        assert 'values' in input_shape
        assert 'query' in input_shape

        # Making sure the input_shape["values"] is a list
        if not isinstance(input_shape["values"], list):
            input_shape["values"] = [input_shape["values"]]

        # The default value for attention_dim is the first values' last dimension size
        if self.attention_dim is None:
            self.attention_dim = int(input_shape["values"][0][-1])

        # Create a trainable weight variable for this layer.
        for i, h in enumerate(input_shape["values"]):
            self.W_h.append(
                self.add_weight(name="W_h_{}".format(i),
                                shape=(int(h[-1]), self.attention_dim),  # (lat_i, d3)
                                initializer='uniform',
                                trainable=True)
            )
        self.U_a = self.add_weight(name='U_a',
                                   shape=(int(input_shape["query"][-1]), self.attention_dim),  # (d2, d3)
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=(1, self.attention_dim),  # (1, d3)
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(
            [input_shape["values"][0], input_shape["query"]])  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: {"values": [encoder_output_sequence], "query": decoder_output_sequence}
        """
        assert isinstance(inputs, dict)
        assert 'values' in inputs
        assert 'query' in inputs

        if not isinstance(inputs["values"], list):
            inputs["values"] = [inputs["values"]]

        encoder_out_seq, decoder_out_seq = inputs["values"], inputs["query"]
        if verbose:
            print('encoder_out_seq>', [K.int_shape(eos) for eos in encoder_out_seq])
            print('decoder_out_seq>', K.int_shape(decoder_out_seq))

        # [h . W_h] of shape [(batch_size, hi, d3)]
        W_hi = [K.dot(eos, self.W_h[i]) for i, eos in enumerate(encoder_out_seq)]

        def broadcast_sum(a, b):
            # Going from (batch_size, a1, d3) to (batch_size, a1, 1, d3)
            a = K.expand_dims(a, 2)
            # Going from (batch_size, b1, d3) to (batch_size, 1, b1, d3)
            b = K.expand_dims(b, 1)
            #  Will be broadcast to (batch_size, a1, b1, d3)
            c = a + b
            # Going from (batch_size, a1, b1, d3) to (batch_size, a1*b2, d3)
            cs = K.shape(c)
            new_shape = K.concatenate([
                cs[0:1], cs[1:2] * cs[2:3], cs[3:4]
            ])
            return K.reshape(c, new_shape)

        # (batch_size, h1*h2*...*hn, d3)
        W_hi = functools.reduce(broadcast_sum, W_hi)

        def broadcast_concat(a, b):
            a_shape = K.shape(a)
            b_shape = K.shape(b)

            # Going from (batch_size, a1, lat_a) to (batch_size, a1, 1, lat_a)
            a = K.expand_dims(a, 2)
            # Going from (batch_size, b1, lat_b) to (batch_size, 1, b1, lat_b)
            b = K.expand_dims(b, 1)

            # (batch_size, b1, lat_a)
            a_complement = tf.zeros(K.concatenate([
                a_shape[0:1],
                b_shape[1:2],
                a_shape[2:3],
            ]))
            # (batch_size, a1, lat_b)
            b_complement = tf.zeros(K.concatenate([
                b_shape[0:1],
                a_shape[1:2],
                b_shape[2:3],
            ]))

            # Going from (batch_size, b1, lat_a) to (batch_size, 1, b1, lat_a)
            a_complement = K.expand_dims(a_complement, 1)
            # Going from (batch_size, a1, lat_b) to (batch_size, a1, 1, lat_b)
            b_complement = K.expand_dims(b_complement, 2)

            # Just to broadcast
            a = a + a_complement
            b = b + b_complement

            # (batch_size, a1, b1, lat_a+lat_b)
            c = K.concatenate([a, b])
            c_shape = K.shape(c)
            # (batch_size, a1*b1, lat_a+lat_b)
            r = K.reshape(c, K.concatenate([
                c_shape[0:1],
                c_shape[1:2] * c_shape[2:3],
                c_shape[3:4],
            ]))
            print("r.shape: ", r.shape)
            return r

        # (batch_size, h1*h2*...*hn, lat1+lat2+...+latn) for later
        hiddens_combined = functools.reduce(broadcast_concat, encoder_out_seq)

        if verbose:
            print('wa.s>', K.int_shape(W_hi))

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            # input: (batch_size, latent_dim)
            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Computing sj.Ua """
            # (batch_size, 1, d3)
            U_a_dot_s = K.expand_dims(K.dot(inputs, self.U_a), 1)
            if verbose:
                print('Ua.h>', K.int_shape(U_a_dot_s))

            """ tanh(h.Wa + s.Ua) """
            # (batch_size, h1*h2*...*hn, d3) = (batch_size, h1*h2*...*hn, d3) + (batch_size, 1, d3)
            Wh_plus_Us = K.tanh(W_hi + U_a_dot_s)
            # (batch_size, d3, h1*h2*...*hn)
            Wh_plus_Us = K.permute_dimensions(Wh_plus_Us, (0, 2, 1))
            if verbose:
                print('Wh+Us>', K.int_shape(Wh_plus_Us))

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # (1, batch_size, h1*h2*...*hn) = (1, d3) . (batch_size, d3, h1*h2*...*hn)
            Wh_plus_Us_dot_Va = K.dot(self.V_a, Wh_plus_Us)
            # (batch_size, h1*h2*...*hn)
            e_i = K.squeeze(Wh_plus_Us_dot_Va, 0)
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', K.int_shape(e_i))

            # (batch_size, h1*h2*...*hn)
            return e_i, states

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # (batch_size, lat1+lat2+...+latn) = (batch_size, h1*h2*...*hn, lat1+lat2+...+latn) * (batch_size, h1*h2*...*hn, 1)
            c_i = K.sum(hiddens_combined * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', K.int_shape(c_i))
            return c_i, states

        # (batch_size, enc_seq_len, latent_dim   encoder_out_seq.shape[1]
        fake_state_e = K.zeros_like(K.placeholder(shape=(1, 1)))
        fake_state_c = K.zeros_like(K.placeholder(shape=(1, 1)))

        """ Computing energy outputs """
        # e_outputs: (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        # c_outputs: (batch_size, de_seq_len, lat1+lat2+...+latn)
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        hidden_size_product = functools.reduce(lambda a, b: a * b,
                                               [s[1] for s in input_shape["values"]], 1)
        latent_size_sum = sum([s[-1] for s in input_shape["values"]])

        return [
            tf.TensorShape((input_shape["query"][0], input_shape["query"][1], latent_size_sum)),
            tf.TensorShape((input_shape["query"][0], input_shape["query"][1], hidden_size_product))
        ]

    def get_config(self):
        return {'attention_dim': self.attention_dim}
