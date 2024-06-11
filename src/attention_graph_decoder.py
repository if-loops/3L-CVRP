import tensorflow as tf

from joblib import Parallel, delayed
import numpy as np
import time
from copy import deepcopy


class GraphAttentionDecoder(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_heads, tanh_clipping=10, decode_type=None):

        super().__init__()

        self.output_dim = output_dim
        self.num_heads = num_heads

        self.head_depth = self.output_dim // self.num_heads
        self.dk_mha_decoder = tf.cast(
            self.head_depth, tf.float32
        )  # for decoding in mha_decoder
        self.dk_get_loc_p = tf.cast(
            self.output_dim, tf.float32
        )  # for decoding in mha_decoder

        if self.output_dim % self.num_heads != 0:
            raise ValueError("number of heads must divide d_model=output_dim")

        self.tanh_clipping = tanh_clipping
        self.decode_type = decode_type

        use_bias = False  # original False
        # we split projection matrix Wq into 2 matrices: Wq*[h_c, h_N, D] = Wq_context*h_c + Wq_step_context[h_N, D]
        self.wq_context = tf.keras.layers.Dense(
            self.output_dim, use_bias=use_bias, name="wq_context",
        )  # (d_q_context, output_dim)
        self.wq_step_context = tf.keras.layers.Dense(
            self.output_dim, use_bias=use_bias, name="wq_step_context",
        )  # (d_q_step_context, output_dim)

        # we need two Wk projections since there is MHA followed by 1-head attention - they have different keys K
        self.wk = tf.keras.layers.Dense(
            self.output_dim, use_bias=use_bias, name="wk",
        )  # (d_k, output_dim)
        self.wk_tanh = tf.keras.layers.Dense(
            self.output_dim, use_bias=use_bias, name="wk_tanh",
        )  # (d_k_tanh, output_dim)

        # we dont need Wv projection for 1-head attention: only need attention weights as outputs
        self.wv = tf.keras.layers.Dense(
            self.output_dim, use_bias=use_bias, name="wv",
        )  # (d_v, output_dim)

        # we dont need wq for 1-head tanh attention, since we can absorb it into w_out
        self.w_out = tf.keras.layers.Dense(
            self.output_dim, use_bias=use_bias, name="w_out",
        )  # (d_model, d_model)

        self.truck_load_embed = tf.keras.layers.Dense(
            self.output_dim, name="truck_embed"
        )
        # self.remaining_emb = tf.keras.layers.Dense(self.output_dim, name="remaining_emb")

        # self.problem = TDCVRP

        self.truck_embed1 = tf.keras.layers.Conv2D(
            1, kernel_size=(5, 5), activation="relu", name="truck_embed_cnn"
        )
        self.truck_embed2 = tf.keras.layers.Flatten()
        self.truck_embed3 = tf.keras.layers.Dense(
            self.output_dim, name="truck_embed_dense"
        )

    def truck_cnn_embed(self, x):
        x = self.truck_embed1(x)
        x = self.truck_embed2(x)
        x = self.truck_embed3(x)
        return x

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def split_heads(self, tensor, batch_size):
        """Function for computing attention on several heads simultaneously
        Splits last dimension of a tensor into (num_heads, head_depth).
        Then we transpose it as (batch_size, num_heads, ..., head_depth) so that we can use broadcast
        """
        tensor = tf.reshape(tensor, (batch_size, -1, self.num_heads, self.head_depth))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def _select_node(self, logits):
        """Select next node based on decoding type.
        """

        # assert tf.reduce_all(logits == logits), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            # probs = tf.exp(logits)
            # selected = tf.math.argmax(probs, axis=-1) # (batch_size, 1)
            selected = tf.math.argmax(logits, axis=-1)  # (batch_size, 1)

        elif self.decode_type == "sampling":
            # logits has a shape of (batch_size, 1, n_nodes), we have to squeeze it
            # to (batch_size, n_nodes) since tf.random.categorical requires matrix
            selected = tf.random.categorical(logits[:, 0, :], 1)  # (bach_size,1)
        else:
            assert False, "Unknown decode type"

        return tf.squeeze(selected, axis=-1)  # (bach_size,)

    def _select_node_sequence(self, logits, masks, same):
        """
        Returns a list of suggested actions with decreasing likelihood
        Prioritizes same location packages as defined in Gendreau
        """

        desc_list = np.zeros([masks.shape[0], masks.shape[2]])

        if self.decode_type == "greedy":
            for batch in range(masks.shape[0]):
                desc_list[batch] = [
                    x
                    for _, x in sorted(
                        zip(logits[batch][0].numpy(), np.arange(masks.shape[2]),),
                        reverse=True,
                    )
                ]
        elif self.decode_type == "sampling":
            for batch in range(masks.shape[0]):

                p = np.exp(logits[batch][0].numpy()) + 0.000000000001

                p = p / np.sum(
                    p
                )  # normalize to sum up to 1 (= get rid of rounding errors)

                # if np.sum(p) == 0:  # nothing feasible -> depot trip
                #    continue

                desc_list[batch] = np.random.choice(
                    np.arange(masks.shape[2]), masks.shape[2], replace=False, p=p,
                )

        else:
            assert False, "Unknown decode type"

        return desc_list

    def get_step_context(
        self, state, embeddings, ppo_emb=None, ppo_truck_cnn=None, ppo_truck_load=None
    ):
        """Takes a state and graph embeddings,
           Returns a part [h_N, D] of context vector [h_c, h_N, D],
           that is related to RL Agent last step.
        """
        if ppo_emb == None:
            # index of previous node
            prev_node = state.prev_a  # (batch_size, 1)

            # from embeddings=(batch_size, n_nodes, input_dim) select embeddings of previous nodes
            cur_embedded_node = tf.gather(
                embeddings, tf.cast(prev_node, tf.int32), batch_dims=1
            )  # (batch_size, 1, input_dim)

            truck_info_cnn = state.get_context()
            truck_info_load = state.get_capacity()
            # truck_info = state.get_max_l_perc()
            # remaining_packages = state.packages_remaining()
        else:
            cur_embedded_node = ppo_emb
            truck_info_cnn = ppo_truck_cnn
            truck_info_load = ppo_truck_load

        # add remaining capacity
        step_context = tf.concat(
            [
                cur_embedded_node,
                self.truck_load_embed(truck_info_load),
                self.truck_cnn_embed(truck_info_cnn),
                # self.remaining_emb(remaining_packages)
            ],
            axis=-1,
        )

        return (
            step_context[:, tf.newaxis, :],
            cur_embedded_node,
            truck_info_cnn,
            truck_info_load,
        )  # (batch_size, 1, input_dim + 1)

    def decoder_mha(self, Q, K, V, mask=None):
        """ Computes Multi-Head Attention part of decoder
        Basically, its a part of MHA sublayer, but we cant construct a layer since Q changes in a decoding loop.

        Args:
            mask: a mask for visited nodes,
                has shape (batch_size, seq_len_q, seq_len_k), seq_len_q = 1 for context vector attention in decoder
            Q: query (context vector for decoder)
                    has shape (..., seq_len_q, head_depth) with seq_len_q = 1 for context_vector attention in decoder
            K, V: key, value (projections of nodes embeddings)
                have shape (..., seq_len_k, head_depth), (..., seq_len_v, head_depth),
                                                                with seq_len_k = seq_len_v = n_nodes for decoder
        """

        # batch_size = tf.shape(Q)[0]

        compatibility = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
            self.dk_mha_decoder
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # dk = tf.cast(tf.shape(K)[-1], tf.float32)
        # compatibility = compatibility / tf.math.sqrt(dk)
        # compatibility = compatibility / tf.math.sqrt(self.dk_mha_decoder)

        if mask is not None:

            # we need to reshape mask:
            # (batch_size, seq_len_q, seq_len_k) --> (batch_size, 1, seq_len_q, seq_len_k)
            # so that we will be able to do a broadcast:
            # (batch_size, num_heads, seq_len_q, seq_len_k) + (batch_size, 1, seq_len_q, seq_len_k)

            mask = mask[:, tf.newaxis, :, :]

            # we use tf.where since 0*-np.inf returns nan, but not -np.inf
            compatibility = tf.where(
                mask, tf.ones_like(compatibility) * (-np.inf), compatibility
            )

        compatibility = tf.nn.softmax(
            compatibility, axis=-1
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)
        attention = tf.matmul(
            compatibility, V
        )  # (batch_size, num_heads, seq_len_q, head_depth)

        # transpose back to (batch_size, seq_len_q, num_heads, depth)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        # concatenate heads (last 2 dimensions)
        attention = tf.reshape(
            attention, (self.batch_size, -1, self.output_dim)
        )  # (batch_size, seq_len_q, output_dim)

        output = self.w_out(
            attention
        )  # (batch_size, seq_len_q, output_dim), seq_len_q = 1 for context att in decoder

        return output

    def get_log_p(self, Q, K, mask=None):
        """Single-Head attention sublayer in decoder,
        computes log-probabilities for node selection.

        Args:
            mask: mask for nodes
            Q: query (output of mha layer)
                    has shape (batch_size, seq_len_q, output_dim), seq_len_q = 1 for context attention in decoder
            K: key (projection of node embeddings)
                    has shape  (batch_size, seq_len_k, output_dim), seq_len_k = n_nodes for decoder
        """

        compatibility = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
            self.dk_get_loc_p
        )

        # dk = tf.cast(tf.shape(K)[-1], tf.float32)
        # compatibility = compatibility / tf.math.sqrt(dk)
        # compatibility = compatibility / tf.math.sqrt(self.dk_get_loc_p)

        compatibility = tf.math.tanh(compatibility) * self.tanh_clipping
        self.dep_comp = compatibility[:, 0, 0]
        # log_p = tf.nn.log_softmax(
        #    compatibility, axis=-1
        # )  # (batch_size, seq_len_q, seq_len_k)

        if mask is not None:

            # we dont need to reshape mask like we did in multi-head version:
            # (batch_size, seq_len_q, seq_len_k) --> (batch_size, num_heads, seq_len_q, seq_len_k)
            # since we dont have multiple heads

            # original
            compatibility = tf.where(
                mask, tf.ones_like(compatibility) * (-np.inf), compatibility
            )
            # log_p = tf.where(mask, tf.ones_like(log_p) * (-np.inf), log_p)

        # Original placement
        log_p = tf.nn.log_softmax(
            compatibility, axis=-1
        )  # (batch_size, seq_len_q, seq_len_k)

        return log_p

    def call(self, embeddings, context_vectors, environ, ppo_info=None):
        # embeddings shape = (batch_size, n_nodes, input_dim)
        # context vectors shape = (batch_size, input_dim)
        self.embeddings = embeddings
        self.batch_size = tf.shape(self.embeddings)[0]

        outputs = []
        sequences = []

        state = environ  # self.problem(inputs)

        # we compute some projections (common for each policy step) before decoding loop for efficiency
        K = self.wk(self.embeddings)  # (batch_size, n_nodes, output_dim)
        K_tanh = self.wk_tanh(self.embeddings)  # (batch_size, n_nodes, output_dim)
        V = self.wv(self.embeddings)  # (batch_size, n_nodes, output_dim)
        Q_context = self.wq_context(
            context_vectors[:, tf.newaxis, :]
        )  # (batch_size, 1, output_dim)

        # we dont need to split K_tanh since there is only 1 head; Q will be split in decoding loop
        K = self.split_heads(
            K, self.batch_size
        )  # (batch_size, num_heads, n_nodes, head_depth)
        V = self.split_heads(
            V, self.batch_size
        )  # (batch_size, num_heads, n_nodes, head_depth)

        # Perform decoding steps
        i = 0
        avg_items_packed = 0

        log_ps = list()

        if ppo_info == None:
            state.emb_history = list()
            state.truck_history_cnn = list()
            state.truck_history_load = list()
            state.mask_history = list()
            state.actions_history = list()
            state.actions_logits_history = list()
            state.triple1_history = list()
            state.triple2_history = list()
            state.triple3_history = list()

        env_stepping_time = 0

        while not state.all_finished() or ppo_info != None:

            if ppo_info == None:
                (
                    step_context,
                    cur_embedded_node,
                    truck_info_cnn,
                    truck_info_load,
                ) = self.get_step_context(
                    state, self.embeddings
                )  # (batch_size, 1, input_dim + 1)
            else:
                (
                    step_context,
                    cur_embedded_node,
                    truck_info_cnn,
                    truck_info_load,
                ) = self.get_step_context(
                    state,
                    self.embeddings,
                    ppo_emb=state.emb_history,
                    ppo_truck_cnn=state.truck_history_cnn,
                    ppo_truck_load=state.truck_history_load,
                )  # (batch_size, 1, input_dim + 1)

            Q_step_context = self.wq_step_context(
                step_context
            )  # (batch_size, 1, output_dim)
            Q = Q_context + Q_step_context
            # print("Q_base", Q_context.shape)
            # print("Q_add", Q_step_context.shape)

            # split heads for Q
            Q = self.split_heads(
                Q, self.batch_size
            )  # (batch_size, num_heads, 1, head_depth)

            # get current mask
            if ppo_info == None:
                mask = (
                    state.get_mask()
                )  # (batch_size, 1, n_nodes) with True/False indicating where agent can go
            else:
                mask = state.mask_history == 1  # set to type bool

            # compute MHA decoder vectors for current mask
            mha = self.decoder_mha(Q, K, V, mask)  # (batch_size, 1, output_dim)

            # compute probabilities
            log_p = self.get_log_p(mha, K_tanh, mask)  # (batch_size, 1, n_nodes)

            if ppo_info != None:  # further steps are not necessary
                return log_p[:, 0], None, None

            # next step is to select node
            # selected = self._select_node(log_p)

            # list of preferences for next node
            selected = self._select_node_sequence(log_p, mask, state.state[:, :, 21])

            pre_step = time.time()
            # state.step(selected.numpy())
            _, _, _, _ = state.step(selected)
            env_stepping_time += time.time() - pre_step

            # prevent NaN in case of constraint violation
            # for batch in range(log_p.shape[0]):  # TODO: find better fix
            #   if log_p[batch, 0,0]  < -100:
            #        log_p = tf.Variable(log_p)
            #        log_p[batch,0, 0].assign(0)

            outputs.append(log_p[:, 0, :])
            # sequences.append(selected)
            sequences.append(state.act_chosen)
            log_ps.append(log_p)

            # print("\n", state.act_chosen, selected)

            i += 1

            # avg_items_packed += np.mean((selected.numpy() > 0) * 1)
            avg_items_packed += np.mean((selected > 0) * 1)

            # for PPO (duplicate but easier to implement with naming)
            # obs_history.append(deepcopy(state)) # do not write to env directly! (recursive saving)
            state.emb_history.append(cur_embedded_node)
            state.truck_history_cnn.append(truck_info_cnn)
            state.truck_history_load.append(truck_info_load)
            state.actions_history.append(deepcopy(state.act_chosen))
            state.actions_logits_history.append(deepcopy(log_p[:, 0]))
            state.mask_history.append(mask)
            state.triple1_history.append(state.original_triple[0])
            state.triple2_history.append(state.original_triple[1])
            state.triple3_history.append(state.original_triple[2])

        ## PPO ---
        state.triple1_history = tf.concat(state.triple1_history, axis=0)
        state.triple2_history = tf.concat(state.triple2_history, axis=0)
        state.triple3_history = tf.concat(state.triple3_history, axis=0)
        state.emb_history = tf.concat(state.emb_history, axis=0)
        state.truck_history_cnn = tf.concat(state.truck_history_cnn, axis=0)
        state.truck_history_load = tf.concat(state.truck_history_load, axis=0)
        state.mask_history = tf.concat(state.mask_history, axis=0)
        state.actions_history_tf = tf.concat(state.actions_history, axis=0)
        state.actions_logits_history_tf = tf.concat(
            state.actions_logits_history, axis=0
        )
        state.actions_logits_history_tf = tf.gather_nd(
            state.actions_logits_history_tf,
            tf.cast(state.actions_history_tf[:, None], tf.int32),
            batch_dims=1,
        )
        # PPO end ---

        all_log_ps = np.concatenate(log_ps).ravel()

        # filter away inf (mostly masked ones)
        state.log_p = all_log_ps[np.isfinite(all_log_ps)]
        state.env_stepping_time += env_stepping_time
        if False:
            print(
                "avg_items_packed: ",
                np.round(avg_items_packed, 1),
                "/",
                environ.state.shape[1] - 1,
                " Batches: ",
                state.state.shape[0],
            )

        # Collected lists, return Tensor
        return (
            tf.stack(outputs, 1),
            tf.cast(tf.stack(sequences, 1), tf.float32),
            state,
        )

