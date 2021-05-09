import unittest
import os
# os.chdir(r'C:\Users\Shahin\Documents\School\Skule\Year 4\Winter\CSC401\Neural_Machine_Translation\code')


from a2_dataloader import *
from a2_encoder_decoder import *
# from a2_training_and_testing import *


class Test(unittest.TestCase):

    def test_encoder_get_all_rnn_inputs(self):
        N, V, H, S, I = 10, 5, 4, 12, 20

        F = torch.randint( low = 0, high = V, size = (S,N))
        encoder = Encoder(V, hidden_state_size=H, word_embedding_size = I)
        x = encoder.get_all_rnn_inputs(F)
        #check shape
        assert x.shape == (S,N,I)
        #check Whenever ``s`` exceeds the original length of ``F[s, n]`` (i.e.
        #when ``F[s, n] == self.pad_id``), ``x[s, n, :] == 0.``
        for s in range(0,S):
            for n in range(0,N):
                if(F[s, n] == encoder.pad_id):
                    assert torch.all(x[s, n, :] == 0)

    def test_encoder_get_all_hidden_states(self):
        N, V, H, S, I = 10, 5, 4, 12, 20
        encoder = Encoder(V, hidden_state_size=H, word_embedding_size = I, cell_type = 'rnn')

        #`x` has been padded such that ``x[F_lens[n]:, n, :] == 0.`` for all ``n``
        x = torch.rand( (S,N,I))
        #make sure F_lens has an element that expands to size S
        F_lens = torch.randint(low = 1, high = S, size = (N,1)).flatten()
        F_lens[-1] = S
        for n in range(0,N):
            x[F_lens[n]:, n, :] = 0

        h_pad = 100

        h = encoder.get_all_hidden_states(x, F_lens, h_pad)
        assert h.shape == (S,N,2 * encoder.hidden_state_size)

        #If``x[s,n] == 0.``, then ``h[s,n, :] == h_pad``
        for s in range(0,S):
            for n in range(0,N):
                if(torch.all(x[s, n] == 0)):
                    assert torch.all(h[s, n, :] == h_pad)


    def test_decoder_wo_attention_get_first_hidden_state(self):
        N, V, H, S = 1, 5, 4, 3
        #make sure F_lens has an element that expands to size S
        F_lens = torch.randint(low = 1, high = S, size = (N,1)).flatten()

        F_lens[-1] = S

        h = torch.rand(size = (S,N,2*H))

        decoder = DecoderWithoutAttention(V, hidden_state_size=2*H)
        htilde_0 = decoder.get_first_hidden_state(h, F_lens)
        assert htilde_0.shape == (N,decoder.hidden_state_size)

        for n in range(0,N):
            seq_size = F_lens[n]
            exp_for = h[seq_size - 1, n, :H]
            act_for = htilde_0[n, :H]

            assert torch.allclose(exp_for, act_for)

            exp_back = h[0, n, H:]
            act_back = htilde_0[n, H:]

            assert torch.allclose(exp_back, act_back)


    def test_decoder_wo_attention_get_current_rnn_input(self):
        N, V, H, S = 10, 5, 4, 12
        #make sure F_lens has an element that expands to size S
        F_lens = torch.randint(low = 1, high = S, size = (N,1)).flatten()
        F_lens[-1] = S
        E_tm1 = torch.randint(low = 1, high = V, size = (N,1)).flatten()
        htilde_tm1 = torch.rand(size = (N,H))
        h = torch.rand(size = (S, N, H))


        decoder = DecoderWithoutAttention(V, hidden_state_size=2*H)
        xtilde_t = decoder.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)
        assert xtilde_t.shape == (N,decoder.word_embedding_size)

        for n in range(0,N):
            if(E_tm1[n] == decoder.pad_id):
                assert torch.all(xtilde_t[n] == 0)


    def test_decoder_wo_attention_get_current_hidden_state(self):
        N, V, H, S, I = 10, 5, 4, 12, 100

        xtilde_t = torch.rand(size = (N,I))
        htilde_tm1 = torch.rand(size = (N, 2*H))


        decoder = DecoderWithoutAttention(V, hidden_state_size=2*H, cell_type = 'rnn', word_embedding_size = I)
        htilde_t = decoder.get_current_hidden_state(xtilde_t, htilde_tm1)
        assert htilde_t.shape == htilde_tm1.shape


    def test_decoder_wo_attention_get_current_logits(self):
        N, V, H, S, I = 10, 5, 4, 12, 100
        htilde_t = torch.rand(size = (N, 2*H))


        decoder = DecoderWithoutAttention(V, hidden_state_size=2*H, cell_type = 'rnn', word_embedding_size = I)
        logits_t = decoder.get_current_logits(htilde_t)
        assert logits_t.shape == (N, V)

    '''
    Insert 
    With attention stuff
    here
    '''


    def test_enc_dec_get_logits_for_teacher_forcing(self):
        N, V, H, S, I, T = 10, 5, 4, 12, 100, 14
        h = torch.rand(size = (S,N, 2*H))
        #make sure F_lens has an element that expands to size S
        F_lens = torch.randint(low = 1, high = S, size = (N,1)).flatten()
        F_lens[-1] = S
        E = torch.randint(low = 1, high = V, size = (T,N))
        enc_dec = EncoderDecoder(Encoder, DecoderWithoutAttention, encoder_hidden_size= H, cell_type = 'rnn', word_embedding_size = I, target_vocab_size = V, source_vocab_size = V)
        E[0,:] = enc_dec.target_eos



        logits_t = enc_dec.get_logits_for_teacher_forcing(h, F_lens, E)
        assert logits_t.shape == (T - 1, N, V)


    # def test_decoder_wo_attention_get_first():
    #     N, V, H, S = 10, 5, 4, 12
    #     h, F_lens = ... # initialize encoder hidden state sequence. No need to *actually* call encoder
    #     htilde_tm1_exp = ... # based on our setup, what should the output look like?
    #     decoder = DecoderWithoutAttention(V, hidden_state_size=2 * H)
    #     htilde_tm1_act = decoder.get_first_hidden_state(h, F_lens)
    #     assert torch.allclose(htilde_tm1_exp, htilde_tm1_act)

if __name__ == '__main__':
    unittest.main( exit = False)
    print('done')