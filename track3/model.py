#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.inference.classifiers import EncoderClassifier

class Projection(nn.Module):
    """
    Simplified Projection module mapping the interaction vector to a final MOS scalar.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        activation=nn.ReLU,
        range_clipping: bool = False,
    ):
        super(Projection, self).__init__()
        self.range_clipping = range_clipping
        
        if self.range_clipping:
            self.proj = nn.Tanh()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        output = self.net(x)

        if self.range_clipping:
            # Scale Tanh [-1, 1] to MOS [1, 5]
            return self.proj(output) * 2.0 + 3
        else:
            return output


class SpeechEncoder(torch.nn.Module):
    """Core feature extractor: Outputs a normalized 1D embedding."""
    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        embedding_dim: int = 256,
        use_projection: bool = True,
        freeze_ssl: bool = False,
    ):
        super().__init__()
        
        # Load to CPU first; PyTorch's Trainer will automatically move it to GPU later
        self.ssl_model = EncoderClassifier.from_hparams(source=model_name, run_opts={"device": "cpu"})
        
        if freeze_ssl:
            for param in self.ssl_model.parameters():
                param.requires_grad = False
        
        # Dynamically determine the embedding dimension using a dummy tensor
        with torch.no_grad():
            dummy_wav = torch.zeros(1, 16000)
            dummy_out = self.ssl_model.encode_batch(dummy_wav)
            output_dim = dummy_out.shape[-1]

        # In zero-shot, we do not want an uninitialized Linear layer scrambling our embeddings
        if use_projection:
            self.projection = nn.Linear(output_dim, embedding_dim)
            self.final_dim = embedding_dim
        else:
            self.projection = nn.Identity()
            self.final_dim = output_dim

    def forward(self, waveform, lengths=None):
        device = waveform.device
        
        # SpeechBrain expects lengths as relative percentages (0.0 to 1.0)
        wav_lens = None
        if lengths is not None:
            wav_lens = (lengths.float() / waveform.shape[1]).to(device)
            
        # Fix SpeechBrain's internal device tracking:
        self.ssl_model.device = device
            
        # encode_batch outputs shape: [Batch, 1, EmbeddingDim]
        embeddings = self.ssl_model.encode_batch(waveform, wav_lens=wav_lens)
        
        # Squeeze out the extra dimension to [Batch, EmbeddingDim]
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        projected_embeddings = self.projection(embeddings)
        normalized_embeddings = F.normalize(projected_embeddings, p=2, dim=-1)
        
        return normalized_embeddings


class Model(torch.nn.Module):
    """
    Speech Similarity Model.
    """
    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        embedding_dim: int = 256,
        use_projection: bool = True,
        freeze_ssl: bool = False,
        mlp_heads: list = None,
        mlp_dnn_dim: int = 64,
        mlp_range_clipping: bool = True,
    ):
        super().__init__()
        self.encoder = SpeechEncoder(model_name, embedding_dim, use_projection, freeze_ssl)
        
        self.mlp_heads = nn.ModuleDict()
        
        if mlp_heads is not None:
            # Interaction dimension uses emb_a, emb_b, abs(emb_a - emb_b), and emb_a * emb_b
            interaction_dim = self.encoder.final_dim * 4
            for head_name in mlp_heads:
                self.mlp_heads[head_name] = Projection(
                    in_dim=interaction_dim,
                    hidden_dim=mlp_dnn_dim,
                    activation=nn.ReLU,
                    range_clipping=mlp_range_clipping,
                )

    def forward(self, wav_a, wav_b=None, len_a=None, len_b=None):
        outputs = {}
        
        emb_a = self.encoder(wav_a, len_a)
        outputs["emb_a"] = emb_a
        
        if wav_b is not None:
            emb_b = self.encoder(wav_b, len_b)
            outputs["emb_b"] = emb_b
            
            outputs["cos_sim"] = F.cosine_similarity(emb_a, emb_b, dim=-1)

            if len(self.mlp_heads) > 0:
                interaction = torch.cat([
                    emb_a, 
                    emb_b,
                    torch.abs(emb_a - emb_b),
                    emb_a * emb_b
                ], dim=-1)
                
                for head_name, mlp in self.mlp_heads.items():
                    head_output = mlp(interaction)
                    outputs[head_name] = head_output.squeeze(-1)

        return outputs