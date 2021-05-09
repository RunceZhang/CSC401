from torch import nn
import torch
from a2_encoder_decoder import DecoderWithAttention


class DecoderSingleDotProductAttention(DecoderWithAttention):
    '''A decoder, this time with scaled dot product attention attention
    Inherits from DecoderWithoutAttention to avoid repeated code.

    Reference: http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/slides/lec16.pdf
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

    def get_energy_scores(self, htilde_t, h):

        # The scale is set to be the 1/dim of the hidden state size
        # Recall:
        #   htilde_t is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   e_t (output) is of shape (S, M)

        # Transform htilde_t to 1 * M * 2H
        if self.cell_type == "lstm":
            htilde_t_temp = htilde_t[0].unsqueeze(1)
        else:
            htilde_t_temp = htilde_t.unsqueeze(1)

        scale = 1/(h.shape[2]**0.5)
        h_temp = h.permute(1, 2, 0)

        e_t = torch.matmul(htilde_t_temp, h_temp) * scale
        return e_t.squeeze(1).transpose(0, 1)


class DecoderMultiDotProductHeadAttention(DecoderSingleDotProductAttention):
    '''A decoder, this time with multi scaled dot product attention
    Inherits from DecoderSingleDotProductionAttention to avoid repeated code.
    '''
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
    '''A decoder using multiplicative attention
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

