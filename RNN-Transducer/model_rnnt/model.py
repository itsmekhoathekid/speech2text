import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, autograd

class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc, dec):

        # enc, dec must both be [B, T, U, C]
        # print(enc.dim())
        # print(dec.dim())
        # # assert enc.shape == dec.shape, f"Shape mismatch: enc={enc.shape}, dec={dec.shape}"
        # if enc.dim() == 3 and dec.dim() == 3:
        #     B, T, C = enc.shape  # enc: [B, T, C]
        #     _, U, _ = dec.shape  # dec: [B, U, C]

        #     # reshape để broadcast
        #     enc = enc.unsqueeze(2).expand(B, T, U, C)  # [B, T, U, C]
        #     dec = dec.unsqueeze(1).expand(B, T, U, C)  # [B, T, U, C]
        
        joint = torch.cat((enc, dec), dim=-1)  # [B, T, U, 2C]
        out = self.forward_layer(joint)
        out = self.tanh(out)
        out = self.project_layer(out)
        return out  # [B, T, U, V]

import k2
class Transducer(nn.Module):
    def __init__(self, encoder, decoder, input_size, inner_dim, vocab_size):
        super(Transducer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.input_size = input_size
        self.inner_dim = inner_dim
        self.vocab_size = vocab_size

        self.joint = JointNet(input_size=self.input_size, inner_dim=self.inner_dim, vocab_size=self.vocab_size)

    def forward(self, inputs, inputs_lengths, targets, targets_lengths):
        zero = torch.zeros((targets.shape[0], 1)).long()
        if targets.is_cuda: zero = zero.cuda()
        
        targets_add_blank = torch.cat((zero, targets), dim=1)
        enc_state, _ = self.encoder(inputs, inputs_lengths)
        dec_state, _ = self.decoder(targets_add_blank, targets_lengths + 1)

        boundary = torch.zeros((inputs.size(0), 4), dtype=torch.int64, device=inputs.device)
        boundary[:, 2] = targets_lengths
        boundary[:, 3] = inputs_lengths

        blank_id = 0
        lm_scale = 0.0
        am_scale = 0.0
        reduction = "mean"
        prune_range = 5

        print(1)
        targets = targets.to(torch.int64)
        simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
            lm=dec_state,
            am=enc_state,
            symbols=targets,
            termination_symbol=blank_id,
            lm_only_scale=lm_scale,
            am_only_scale=am_scale,
            boundary=boundary,
            reduction=reduction,
            return_grad=True,
        )
        print(2)
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )
        print(3)
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=enc_state, lm=dec_state, ranges=ranges
        )

        logits = self.joint(am_pruned, lm_pruned)

        print(4)
        pruned_loss = k2.rnnt_loss_pruned(
            logits=logits,
            symbols=targets,
            ranges=ranges,
            termination_symbol=blank_id,
            boundary=boundary,
            reduction=reduction,
        )

        return logits, pruned_loss

    #only one
    def recognize(self, inputs, inputs_length):
    
        batch_size = inputs.size(0)

        enc_states, _ = self.encoder(inputs, inputs_length)

        zero_token = torch.LongTensor([[0]])

        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []
            dec_state, hidden = self.decoder(zero_token)

            #print(len(hidden))
            for t in range(lengths):
                logits = self.joint(enc_state[t].view(1, 1, 1, -1), dec_state.view(1, 1, 1, -1))[0, 0, 0]
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0)
                pred = int(pred.item())

                if pred != 0:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])

                    if enc_state.is_cuda:
                        token = token.cuda()

                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list

        results = []

        for i in range(batch_size):
            decoded_seq = decode(enc_states[i], inputs_length[i])
            results.append(decoded_seq)

        return results

    def beam_search(self, inputs, inputs_length, W): 
        use_gpu = inputs.is_cuda
        def isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b): return False
            for i in range(len(a)):
                if a[i] != b[i]: return False
            return True

        def forward_step(label, hidden):
            #if use_gpu: label = label.cuda()
            
            label = torch.LongTensor([[label]])

            if use_gpu: label = label.cuda()

            
            pred, hidden = self.decoder(inputs=label, hidden = hidden)

            return pred[0][0], hidden
        
        B = [Sequence(blank=0)]
        
        batch_size = inputs.size(0)
        enc_states, _ = self.encoder(inputs, inputs_length)

        enc_states_for_beam = enc_states.squeeze()
        
        prefix = False

        for i, x in enumerate(enc_states_for_beam):
            sorted(B, key=lambda a: len(a.k), reverse=True)
            A = B
            B = []

            if prefix:
                for j in range(len(A)-1):
                    for i in range(j+1, len(A)):
                        if not isprefix(A[i].k, A[j].k): continue
                        
                        pred, _ = forward_step(A[i].k[-1], A[i].h)
                        ytu = self.joint(x, pred)
                        logp = F.log_softmax(ytu, dim=0)
                        curlogp = A[i].logp + float(logp[A[j].k[idx]])
                        for k in range(idx, len(A[j].k)-1):
                            ytu = self.joint(x, A[j].g[k])
                            logp = F.log_softmax(ytu, dim=0)
                            curlogp += float(logp[A[j].k[k+1]])
                        A[j].logp = log_aplusb(A[j].logp, curlogp)

            while True:
                y_hat = max(A, key=lambda a: a.logp)
                A.remove(y_hat)
                #print(y_hat.k) #첫번째 0
                #print(y_hat.h) #첫번째 none
                
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h)
                ytu = self.joint(x, pred)

                logp = F.log_softmax(ytu, dim=0)
                
                for k in range(len(logp)):
                    yk = Sequence(y_hat)

                    yk.logp += float(logp[k])

                    if k == 0:
                        B.append(yk)                        
                        continue

                    yk.h = hidden; yk.k.append(k); 
                    
                    
                    if prefix: yk.g.append(pred)
                    
                    A.append(yk)

                y_hat = max(A, key=lambda a: a.logp)               
                yb = max(B, key=lambda a: a.logp)

                if len(B) >= W and yb.logp >= y_hat.logp: break
                
            sorted(B, key=lambda a: a.logp, reverse=True)

        return B[0].k, -B[0].logp

class Sequence():
    def __init__(self, seq=None, blank=0):
        if seq is None:
            self.g = [] # predictions of phoneme language model
            self.k = [blank] # prediction phoneme label
            # self.h = [None] # input hidden vector to phoneme model
            self.h = None
            self.logp = 0 # probability of this sequence, in log scale
        else:
            self.g = seq.g[:] # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp


