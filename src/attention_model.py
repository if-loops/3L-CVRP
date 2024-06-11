import tensorflow as tf


from attention_graph_encoder import GraphAttentionEncoder
from attention_graph_decoder import GraphAttentionDecoder
import numpy as np
import time

# from envs.Environment import TDCVRP


def set_decode_type(model, decode_type):
    model.set_decode_type(decode_type)
    model.decoder.set_decode_type(decode_type)


class AttentionModel(tf.keras.Model):
    def __init__(self, embedding_dim, n_encode_layers=3, n_heads=8, tanh_clipping=10.0):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        # self.problem = TDCVRP
        self.n_heads = n_heads

        self.embedder = GraphAttentionEncoder(
            input_dim=self.embedding_dim,
            num_heads=self.n_heads,
            num_layers=self.n_encode_layers,
        )

        self.decoder = GraphAttentionDecoder(
            num_heads=self.n_heads,
            output_dim=self.embedding_dim,
            tanh_clipping=tanh_clipping,
            decode_type=self.decode_type,
        )

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def _calc_log_likelihood(self, _log_p, a):

        # Get log_p corresponding to selected actions
        log_p = tf.gather_nd(
            _log_p, tf.cast(tf.expand_dims(a, axis=-1), tf.int32), batch_dims=2
        )

        # mask already done in decoder log_p step

        assert (
            log_p.numpy() + 10001
        ).all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return tf.reduce_sum(log_p, 1)

    def call(self, in_env=None, return_pi=False, usage=False, ppo_info=None):

        start_time = time.time()

        if ppo_info == None:
            triple_0, triple_1, triple_2 = in_env.get_triple()
            triple_0 = tf.convert_to_tensor(triple_0, np.float32)
            triple_1 = tf.convert_to_tensor(triple_1, np.float32)
            triple_2 = tf.convert_to_tensor(triple_2, np.float32)
            in_env.original_triple = [triple_0, triple_1, triple_2]  # for ppo passing
        else:
            triple_0 = (
                in_env.triple1_history
            )  # TODO: fix naming as it is one off currently
            triple_1 = in_env.triple2_history
            triple_2 = in_env.triple3_history

        # print([triple_0, triple_1, triple_2])
        embeddings, mean_graph_emb = self.embedder([triple_0, triple_1, triple_2])
        # print("OUT", embeddings, mean_graph_emb)

        _log_p, pi, out_env = self.decoder(
            embeddings=embeddings,
            context_vectors=mean_graph_emb,
            environ=in_env,
            ppo_info=ppo_info,
        )

        if ppo_info != None:
            return _log_p

        cost = tf.convert_to_tensor(out_env.get_costs(), dtype=tf.float32)
        out_env.model_time += time.time() - start_time

        ll = self._calc_log_likelihood(_log_p, pi)

        if usage:
            return cost, ll, out_env

        if return_pi:
            return cost, ll, pi

        return cost, ll
