'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall

All of the files in this directory and all subdirectories are:
Copyright (c) 2021 University of Toronto
'''

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch
from torch import nn

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase

# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.
class Encoder(EncoderBase):


    def init_submodules(self):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.rnn, self.embedding
        # 2. You will need these object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, GRU, RNN, Embedding}

        models = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
        self.embedding = nn.Embedding(num_embeddings=self.source_vocab_size,
                                      embedding_dim=self.word_embedding_size,
                                      padding_idx=self.pad_id)

        self.rnn = models[self.cell_type](
            input_size=self.word_embedding_size,
            hidden_size=self.hidden_state_size,
            num_layers=self.num_hidden_layers,
            dropout=self.dropout,
            bidirectional=True # BERT
        )

    def forward_pass(self, F, F_lens, h_pad=0.):
        # Recall:
        #   F is shape (S, M)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use these methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states

        F_emb = self.get_all_rnn_inputs(F)
        return self.get_all_hidden_states(F_emb, F_lens, h_pad=h_pad)

    def get_all_rnn_inputs(self, F):
        # Recall:
        #   F is shape (S, M)
        #   x (output) is shape (S, M, I)
        return self.embedding(F)

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # Recall:
        #   x is of shape (S, M, I)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #   h (output) is of shape (S, M, 2 * H)

        # Avoiding computing 'Pads', which are added to accommodate for variable-length input
        # Packing saves computation cost by tracking actual batch size at each time step
        x_packed = nn.utils.rnn.pack_padded_sequence(input=x, # padded batch of sequence (3-dim)
                                                     lengths=F_lens, # Sequence lengths of each batch element
                                                     enforce_sorted=False)

        # h_packed (last layer of RNN output at timestep S, of the Mth sequence in the batch)
        if self.cell_type == "lstm":
            h_packed, (hn, cn) = self.rnn(x_packed)
        else:
            h_packed, hn = self.rnn(x_packed)
        h_padded, len_padded = nn.utils.rnn.pad_packed_sequence(h_packed, padding_value=h_pad)
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        return h_padded


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.embedding, self.cell, self.ff
        # 2. You will need these object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        self.embedding = nn.Embedding(num_embeddings=self.target_vocab_size,
                                      embedding_dim=self.word_embedding_size,
                                      padding_idx=self.pad_id)

        cells = {"lstm": nn.LSTMCell, "gru": nn.GRUCell, "rnn": nn.RNNCell}
        self.cell = cells[self.cell_type](
            input_size=self.word_embedding_size, # Number of features in input
            hidden_size=self.hidden_state_size, # Number of features in hidden state
        )
        self.ff = nn.Linear(in_features=self.hidden_state_size,
                            out_features=self.target_vocab_size)

    def forward_pass(self, E_tm1, htilde_tm1, h, F_lens):
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   logits_t (output) is of shape (M, V)
        #   htilde_t (output) is of same shape as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use these methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.

        # Obtain encoder RNN input
        xtilde_t = self.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)

        # Find decoder hidden states
        htilde_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)

        # Un-normalized log-probabilities over target vocab
        logits_t = self.get_current_logits(htilde_t)
        return logits_t, htilde_t


    def get_first_hidden_state(self, h, F_lens):
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   htilde_tm1 (output) is of shape (M, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch functions: torch.cat

        # highest-t in forward direction (avoid padded values by slicing on F_lens)
        forward = h[F_lens - 1, [_ for _ in range(F_lens.shape[0])], :self.hidden_state_size//2]

        # t=0 in backward direction
        backward = h[0, :, self.hidden_state_size//2:]

        # Concatenate forward and backward
        htilde_tm1 = torch.cat((forward, backward), dim=1).to(h.device)
        return htilde_tm1


    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   xtilde_t (output) is of shape (M, Itilde)

        # Current RNN Input
        # # Initialize shape
        # xtilde_t = torch.zeros(F_lens.shape[0], self.word_embedding_size)
        #
        # # Case when Etm1 = self.pad_id, set xtilde_t[m] == 0
        # xtilde_t[E_tm1 == self.pad_id] = 0
        #
        # # Embed on those that Etm1 != self.pad_id
        # xtilde_t[E_tm1 != self.pad_id] = self.embedding(E_tm1[E_tm1 != self.pad_id])
        return self.embedding(E_tm1)


    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # Recall:
        #   xtilde_t is of shape (M, Itilde)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same shape as htilde_tm1

        return self.cell(xtilde_t, htilde_tm1)

    def get_current_logits(self, htilde_t):
        # Recall:
        #   htilde_t is of shape (M, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of shape (M, V)

        # Case lstm: only pass in the first element in the tuple
        if self.cell_type == "lstm":
            logits_t = self.ff(htilde_t[0])
        else:
            logits_t = self.ff(htilde_t)
        return logits_t


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need these object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.


        cells = {"lstm": nn.LSTMCell, "gru": nn.GRUCell, "rnn": nn.RNNCell}

        # Modify input size to accomodate for context vector
        self.cell = cells[self.cell_type](
            input_size=self.word_embedding_size + self.hidden_state_size,
            hidden_size=self.hidden_state_size)

        self.embedding = nn.Embedding(num_embeddings=self.target_vocab_size,
                                      embedding_dim=self.word_embedding_size,
                                      padding_idx=self.pad_id)
        self.ff = nn.Linear(in_features=self.hidden_state_size,
                            out_features=self.target_vocab_size)



    def get_first_hidden_state(self, h, F_lens):
        # Hint: For this time, the hidden states should be initialized to zeros.

        # Using zeros_like to accomodate for certain corner cases
        return torch.zeros_like(h[0]).to(h.device)

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # Hint: Use attend() for c_t
        xtilde_t = self.embedding(E_tm1)

        # Find context
        c_tm1 = self.attend(htilde_tm1, h, F_lens)

        # Concatenate embedded and context vector
        xtilde_t = torch.cat((xtilde_t, c_tm1), dim=1).to(h.device)
        return xtilde_t


    def attend(self, htilde_t, h, F_lens):
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)``. The
            context vectorc_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''

        # Dimension: S * M
        alpha_t = self.get_attention_weights(htilde_t, h, F_lens)

        # Dimension after squeezing: S * 1 * M, after transposing: M * 1 * S
        alpha_trans = alpha_t.unsqueeze(dim=1).transpose(0, 2)

        # Dimension after transposing: M * S * 2H
        h_trans = h.transpose(0, 1)

        # Matmul performs a batched multiplication using 1st dimension as the batch
        # Dimension: M * 1 * S @ M * S * 2H, after matmul:  M * 1 * 2H, after squeeze: M * 2H
        return torch.matmul(alpha_trans, h_trans).squeeze(1)

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of shape (S, M)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, M)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Recall:
        #   htilde_t is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   e_t (output) is of shape (S, M)
        #
        # Hint:
        # Relevant pytorch functions: torch.nn.functional.cosine_similarity

        # LSTM: use only 1st tuple
        if self.cell_type == "lstm":
            e_t = nn.CosineSimilarity(dim=-1)(htilde_t[0], h)
        else:
            e_t = nn.CosineSimilarity(dim=-1)(htilde_t, h)
        return e_t

class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not modify this line

        # Hints:
        # 1. The above line should ensure self.ff, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize these submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need these object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. You do *NOT* need self.heads at this point
        # 6. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)

        # Size for all submodules are H * H (Does not depend on the number of heads)
        self.W = nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)
        self.Wtilde = nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)
        self.Q = nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)

    def attend(self, htilde_t, h, F_lens):
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch functions:
        #   tensor().repeat_interleave, tensor().view
        # 3. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        # 4. You *WILL* need self.heads at this point

        # Case LSTM: Use only 2nd tuple
        # Let K represent the number of heads
        # Transformation: htilde_t_n = MK * H/K, after applying linear layer
        # To ensure that the energy_score function is computed correctly, we can only expand along the batch dimension
        if self.cell_type == "lstm":
            htilde_t_n = (self.Wtilde(htilde_t[0]).view(h.shape[1] * self.heads, h.shape[2]//self.heads),
                          htilde_t[1].view(h.shape[1] * self.heads, h.shape[2]//self.heads))
        else:
            htilde_t_n = self.Wtilde(htilde_t).view(h.shape[1] * self.heads, h.shape[2]//self.heads)

        # Transformation: h_n = S * MK * H/K, after applying linear layer
        h_n = self.W(h).view(h.shape[0], h.shape[1] * self.heads, h.shape[2]//self.heads)

        # Transformation: F_lens_n = MK
        # This ensures that padding is applied correctly as F_lens_n keeps track of the last element in sequence
        F_lens_n = F_lens.repeat_interleave(self.heads)

        # Call attend (for single-head) and transform from MK * H/K to M * H context vector
        c_t_n = super().attend(htilde_t_n, h_n, F_lens_n).view(h.shape[1], h.shape[2])

        # Apply a final linear layer
        return self.Q(c_t_n)

class DecoderMultiplicativeAttention(DecoderWithAttention):
    '''A decoder, this time with scaled dot product attention attention
    Inherits from DecoderWithoutAttention to avoid repeated code.

    Reference: https://ruder.io/deep-learning-nlp-best-practices/
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.Wm is not None, 'initialize Wm!'

    def init_submodules(self):
        super().init_submodules()  # Do not modify this line

        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need these object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.

        self.Wm = nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)

    def get_energy_scores(self, htilde_t, h):

        # Recall:
        #   htilde_t is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   e_t (output) is of shape (S, M)

        # Apply a Layer on h_trans -> M * 2H * S
        h_temp = self.Wm(h).permute(1, 2, 0)

        # M * 1 * 2H
        if self.cell_type == "lstm":
            htilde_t_temp = htilde_t[0].unsqueeze(1)
        else:
            htilde_t_temp = htilde_t.unsqueeze(1)

        # Scale is the hidden state size
        scale = 1/(h.shape[2]**0.5)

        # Find dot product of each, multiplied by a scale
        e_t = torch.matmul(htilde_t_temp, h_temp) * scale
        return e_t.squeeze(1).transpose(0, 1)


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need these object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it

        self.encoder = encoder_class(source_vocab_size=self.source_vocab_size,
                                     pad_id=self.source_pad_id,
                                     word_embedding_size=self.word_embedding_size,
                                     num_hidden_layers=self.encoder_num_hidden_layers,
                                     hidden_state_size=self.encoder_hidden_size,
                                     dropout=self.encoder_dropout,
                                     cell_type=self.cell_type)

        self.decoder = decoder_class(target_vocab_size=self.target_vocab_size,
                                     pad_id=self.target_eos,
                                     word_embedding_size=self.word_embedding_size,
                                     hidden_state_size=self.encoder_hidden_size * 2,
                                     cell_type=self.cell_type,
                                     heads=self.heads)

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   E is of shape (T, M)
        #   logits (output) is of shape (T - 1, M, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than E (why?)

        # First decoder hidden state not initialized
        htilde_t = None

        # Logits size
        logits = torch.zeros(E.shape[0] - 1, h.shape[1], self.target_vocab_size).to(h.device)

        # Iterate through timestep
        for t in range(1, E.shape[0]):
            htilde_tm1 = htilde_t

            # Inputs are Etm1 and htilde_tm1
            logits_t, htilde_t = self.decoder.forward(E[t - 1, :], htilde_tm1, h, F_lens)
            logits[t - 1, :, :] = logits_t

        logits = logits.to(h.device)
        return logits


    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of shape (M, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of shape (M, K)
        #   b_tm1_1 is of shape (t, M, K)
        #   b_t_0 (first output) is of shape (M, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (second output) is of shape (t + 1, M, K)
        #   logpb_t (third output) is of shape (M, K)
        #
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of shape z of (A, B),
        #   then the element z[a, b] maps to z'[a*B + b]

        # Transformation: M * K to M * K * 1 to M * K * V
        logpb_tm1_repeat = logpb_tm1.unsqueeze(-1).repeat(1, 1, self.target_vocab_size)

        # Transformation: M * K * V to M * KV
        extensions_t = (logpb_tm1_repeat + logpy_t).reshape(logpy_t.shape[0], logpy_t.shape[1] * logpy_t.shape[2])

        # Pick top K paths (identical to greedy if K = 1)
        # v indexes the maximal k elements in dim=1 of extensions_t
        # logpb_t beam search prefixes up to time t
        logpb_t, v = extensions_t.topk(logpb_tm1.shape[1], dim=1)

        # Find the prefixes that are kept and the words that can be extended
        valid_prefixes = torch.div(v, self.target_vocab_size).unsqueeze(0)
        next_words = torch.remainder(v, self.target_vocab_size).unsqueeze(0)

        # Keep only the valid paths in b_tm1_1
        b_tm1_1 = b_tm1_1.gather(2, valid_prefixes.expand_as(b_tm1_1))

        # Concatenate the next word to b_t_1
        b_t_1 = torch.cat([b_tm1_1, next_words], dim=0)

        # Transformation: 1 * M * K to M * K * 1
        valid_prefixes = valid_prefixes.reshape(valid_prefixes.shape[1], valid_prefixes.shape[2], valid_prefixes.shape[0])

        # Update b_t_0 to have the valid prefixes only
        if self.cell_type == "lstm":
            b_t_0  = (htilde_t[0].gather(1, valid_prefixes.expand_as(htilde_t[0])),
                      htilde_t[1].gather(1, valid_prefixes.expand_as(htilde_t[1])))
        else:
            b_t_0 = htilde_t.gather(1, valid_prefixes.expand_as(htilde_t))
        return b_t_0, b_t_1, logpb_t