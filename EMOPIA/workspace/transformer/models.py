
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

# Original EMOPIA Sampling Logic (Ported to PyTorch)
def softmax_with_temperature(logits, temperature):
    # Stabilize by subtracting max
    logits = logits - torch.max(logits)
    probs = torch.exp(logits / temperature)
    probs = probs / torch.sum(probs)
    if torch.isnan(probs).any():
        return None
    return probs

def nucleus(probs, p):
    # probs is a 1D tensor
    sorted_probs, sorted_index = torch.sort(probs, descending=True)
    cusum_sorted_probs = torch.cumsum(sorted_probs, dim=0)
    after_threshold = cusum_sorted_probs > p
    if torch.sum(after_threshold) > 0:
        # Get the first index where cumulative probability exceeds p
        last_index = torch.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    
    candi_probs = probs[candi_index]
    candi_probs = candi_probs / torch.sum(candi_probs)
    
    # Sample from candidate indices
    word_idx = torch.multinomial(candi_probs, 1).item()
    return candi_index[word_idx].item()

def weighted_sampling(probs):
    probs = probs / torch.sum(probs)
    return torch.multinomial(probs, 1).item()

def sampling(logit, p=None, t=1.0, is_training=False):
    if is_training:
        return torch.argmax(logit)
    
    # logit should be 1D
    probs = softmax_with_temperature(logit, t)
    if probs is None:
        return None

    if p is not None:
        return nucleus(probs, p=p)
    else:
        return weighted_sampling(probs)

D_MODEL = 512
N_HEAD = 8
N_LAYER = 12

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LinearAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(LinearAttention, self).__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, memory=None):
        if len(q.shape) == 2: q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        b, s, d = q.shape
        queries = self.query_projection(q).view(b, s, self.n_head, self.d_head)
        keys = self.key_projection(k).view(b, s, self.n_head, self.d_head)
        values = self.value_projection(v).view(b, s, self.n_head, self.d_head)

        # Feature map: ELU + 1 is typical for linear transformer
        queries = F.elu(queries) + 1.0
        keys = F.elu(keys) + 1.0

        if memory is None:
            # Full sequence pass (causality handled via triangular mask or cumsum)
            # For init tokens we can just do global sum as a simplification
            k_v = torch.einsum("bshd,bshm->bhdm", keys, values)
            z = keys.sum(dim=1).unsqueeze(1)
        else:
            prev_k_v, prev_z = memory
            k_v = prev_k_v + torch.einsum("bshd,bshm->bhdm", keys[:, -1:], values[:, -1:])
            z = prev_z + keys[:, -1:]
        
        num = torch.einsum("bshd,bhdm->bshm", queries, k_v)
        den = torch.einsum("bshd,bshd->bsh", queries, z)
        out = num / (den.unsqueeze(-1) + 1e-6)
        
        out = out.reshape(b, s, d)
        return self.out_projection(out), (k_v, z)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super(TransformerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = LinearAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, 2048)
        self.linear2 = nn.Linear(2048, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, memory=None):
        res = x
        x = self.norm1(x)
        x, new_memory = self.attention(x, x, x, memory)
        x = res + self.dropout(x)
        # Hidden state clipping to prevent instability
        x = torch.clamp(x, -100.0, 100.0)
        
        res = x
        x = self.norm2(x)
        x = self.linear2(F.gelu(self.linear1(x)))
        x = res + self.dropout(x)
        x = torch.clamp(x, -100.0, 100.0)
        return x, new_memory

class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, d_model, n_head):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_head) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory=None):
        if memory is None:
            memory = [None] * len(self.layers)
        new_memories = []
        for i, layer in enumerate(self.layers):
            x, m = layer(x, memory[i])
            new_memories.append(m)
        return self.norm(x), new_memories

class TransformerModel(nn.Module):
    def __init__(self, n_token, is_training=True, data_parallel=False):
        super(TransformerModel, self).__init__()
        self.n_token = n_token
        self.d_model = D_MODEL
        self.n_layer = N_LAYER
        self.n_head = N_HEAD
        self.emb_sizes = [128, 256, 64, 32, 512, 128, 128, 128]

        self.word_emb_tempo     = Embeddings(n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = Embeddings(n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = Embeddings(n_token[2], self.emb_sizes[2])
        self.word_emb_type      = Embeddings(n_token[3], self.emb_sizes[3])
        self.word_emb_pitch     = Embeddings(n_token[4], self.emb_sizes[4])
        self.word_emb_duration  = Embeddings(n_token[5], self.emb_sizes[5])
        self.word_emb_velocity  = Embeddings(n_token[6], self.emb_sizes[6])
        self.word_emb_emotion   = Embeddings(n_token[7], self.emb_sizes[7])
        
        self.pos_emb = PositionalEncoding(self.d_model)
        self.in_linear = nn.Linear(1376, self.d_model)
        self.transformer_encoder = TransformerEncoder(self.n_layer, self.d_model, self.n_head)
        self.project_concat_type = nn.Linear(self.d_model + 32, self.d_model)

        self.proj_tempo    = nn.Linear(self.d_model, n_token[0])
        self.proj_chord    = nn.Linear(self.d_model, n_token[1])
        self.proj_barbeat  = nn.Linear(self.d_model, n_token[2])
        self.proj_type     = nn.Linear(self.d_model, n_token[3])
        self.proj_pitch    = nn.Linear(self.d_model, n_token[4])
        self.proj_duration = nn.Linear(self.d_model, n_token[5])
        self.proj_velocity = nn.Linear(self.d_model, n_token[6])
        self.proj_emotion  = nn.Linear(self.d_model, n_token[7])

    def forward_hidden(self, x, memory=None, is_training=False):
        embs = torch.cat([
            self.word_emb_tempo(x[..., 0]),
            self.word_emb_chord(x[..., 1]),
            self.word_emb_barbeat(x[..., 2]),
            self.word_emb_type(x[..., 3]),
            self.word_emb_pitch(x[..., 4]),
            self.word_emb_duration(x[..., 5]),
            self.word_emb_velocity(x[..., 6]),
            self.word_emb_emotion(x[..., 7])
        ], dim=-1)
        # Stabilize embeddings
        embs = torch.clamp(embs, -10.0, 10.0)
        h = self.in_linear(embs)
        h = self.pos_emb(h)
        h, memory = self.transformer_encoder(h, memory)
        if not is_training and h.size(1) == 1:
            h = h.squeeze(1)
        y_type = self.proj_type(h)
        return h, y_type, memory

    def froward_output_sampling(self, h, y_type, is_training=False):
        y_type_logit = y_type[0] if len(y_type.shape) > 1 else y_type
        # Type sampling
        cur_word_type = sampling(y_type_logit, p=0.90, is_training=is_training)
        if cur_word_type is None: return None, None

        type_word_t = torch.tensor([[cur_word_type]]).long().to(h.device)
        tf_skip_type = self.word_emb_type(type_word_t).squeeze(0)
        
        if len(h.shape) == 1: h = h.unsqueeze(0)
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type).squeeze(0)

        curs = [
            sampling(self.proj_tempo(y_), t=1.2, p=0.9, is_training=is_training),
            sampling(self.proj_chord(y_), p=0.99, is_training=is_training),
            sampling(self.proj_barbeat(y_), t=1.2, is_training=is_training),
            sampling(self.proj_pitch(y_), p=0.9, is_training=is_training),
            sampling(self.proj_duration(y_), t=2, p=0.9, is_training=is_training),
            sampling(self.proj_velocity(y_), t=5, is_training=is_training)
        ]
        if None in curs: return None, None
        
        # clamp tempo to slower range
        curs[0] = min(max(curs[0], 30), 90)
        
        # Force barbeat progression to advance MIDI timeline
        if not hasattr(self, 'last_barbeat'):
            self.last_barbeat = 0
        else:
            self.last_barbeat = (self.last_barbeat + 1) % 16
            
        next_barbeat = self.last_barbeat
        
        next_arr = np.array([curs[0], curs[1], next_barbeat, cur_word_type, curs[3], curs[4], curs[5], 0])
        
        # Clamp note duration to prevent extremely short notes
        next_arr[5] = max(next_arr[5], 30)
        
        return next_arr, self.proj_emotion(y_)

    def inference_from_scratch(self, dictionary, emotion_tag, n_token=8, display=True, max_steps=512):
        event2word, word2event = dictionary
        device = next(self.parameters()).device
        target_emotion = [0, 0, 0, 1, 0, 0, 0, emotion_tag]
        bar_init = [0, 0, 1, 2, 0, 0, 0, 0]
        init = np.array([target_emotion, bar_init])
        final_res, memory = [], None
        self.last_barbeat = 0 # Reset progression
        with torch.no_grad():
            for step in range(init.shape[0]):
                token = init[step]
                final_res.append(token[None, ...])
                h, y_type, memory = self.forward_hidden(torch.from_numpy(token).long().to(device).unsqueeze(0).unsqueeze(0), memory)
            for i in range(max_steps):
                # Strict Anti-EOS sampling
                next_arr, _ = self.froward_output_sampling(h, y_type)
                if next_arr is None: break
                
                is_eos = word2event['type'][next_arr[3]] == 'EOS'
                min_steps = int(max_steps * 0.9)
                
                # If hit EOS too early, strictly ignore it and pick a Note
                if is_eos and len(final_res) < min_steps:
                    # Pick a different token. We try to find a 'Note' or 'Metrical'
                    # For simplicity, we sample again until we get a non-EOS
                    retry = 0
                    while is_eos and retry < 50:
                        next_arr, _ = self.froward_output_sampling(h, y_type)
                        if next_arr is None: break
                        is_eos = word2event['type'][next_arr[3]] == 'EOS'
                        retry += 1
                        
                    if is_eos: # Still EOS even after retries, force change
                        # Force type to 'Note' (index 0) or 'Metrical' (1, 2)
                        next_arr[3] = random.choice([0, 1, 2])
                        is_eos = False

                final_res.append(next_arr[None, ...])
                if is_eos: break
                
                h, y_type, memory = self.forward_hidden(torch.from_numpy(next_arr).long().to(device).view(1, 1, 8), memory)
                if len(final_res) >= max_steps: break
        return np.concatenate(final_res), None
