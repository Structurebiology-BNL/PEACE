"""Model definitions for package-native workflows."""

from __future__ import annotations

import torch.nn as nn


class SimplePredictor(nn.Module):
    """Neural network model for protein-level prediction with a shared encoder."""

    def __init__(
        self,
        input_dim=1280,
        output_dim=1,
        dropout_rate=0.2,
        use_contrastive=False,
        contrastive_dim=128,
        encoder_hidden_dim=None,
    ):
        super(SimplePredictor, self).__init__()

        if encoder_hidden_dim is None:
            encoder_hidden_dim = input_dim

        self.encoder_hidden_dim = encoder_hidden_dim
        self.shared_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.classification_head = nn.Sequential(
            nn.LayerNorm(encoder_hidden_dim),
            nn.Linear(encoder_hidden_dim, output_dim),
        )

        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_head = nn.Sequential(
                nn.LayerNorm(encoder_hidden_dim),
                nn.Linear(encoder_hidden_dim, contrastive_dim),
            )

        self.training_mode = "finetuning"

    def set_training_mode(self, mode: str):
        """Set whether the model emits pretraining or finetuning outputs."""
        if mode not in ["pretraining", "finetuning"]:
            raise ValueError(
                f"Training mode must be 'pretraining' or 'finetuning', got {mode}"
            )
        self.training_mode = mode

    def _compute_shared_features(self, features):
        """Compute shared encoder features for 2D or 3D inputs."""
        if features.dim() == 3:
            batch_size, num_variants, input_dim = features.shape
            features_flat = features.view(batch_size * num_variants, input_dim)
            shared_features_flat = self.shared_encoder(features_flat)
            return shared_features_flat.view(
                batch_size,
                num_variants,
                self.encoder_hidden_dim,
            )
        return self.shared_encoder(features)

    def _compute_contrastive_embeddings(self, shared_features):
        """Project shared features into the contrastive embedding space."""
        if shared_features.dim() == 3:
            batch_size, num_variants, encoder_hidden_dim = shared_features.shape
            features_flat = shared_features.view(
                batch_size * num_variants,
                encoder_hidden_dim,
            )
            contrastive_embeddings_flat = self.contrastive_head(features_flat)
            contrastive_dim = contrastive_embeddings_flat.shape[-1]
            return contrastive_embeddings_flat.view(
                batch_size,
                num_variants,
                contrastive_dim,
            )
        return self.contrastive_head(shared_features)

    def forward(self, features, return_features=False):
        """Run the model and emit outputs for the active training mode."""
        shared_features = self._compute_shared_features(features)

        if self.training_mode == "pretraining":
            if not self.use_contrastive:
                raise ValueError("Pretraining mode requires use_contrastive=True")

            contrastive_embeddings = self._compute_contrastive_embeddings(
                shared_features
            )
            if return_features:
                return contrastive_embeddings, shared_features
            return contrastive_embeddings

        if shared_features.dim() == 3:
            shared_features_original = shared_features[:, 0, :]
            logits = self.classification_head(shared_features_original)

            if self.use_contrastive:
                contrastive_embeddings = self._compute_contrastive_embeddings(
                    shared_features
                )
                if return_features:
                    return logits, contrastive_embeddings, shared_features
                return logits, contrastive_embeddings

            if return_features:
                return logits, shared_features
            return logits

        logits = self.classification_head(shared_features)

        if self.use_contrastive:
            contrastive_embeddings = self._compute_contrastive_embeddings(
                shared_features
            )
            if return_features:
                return logits, contrastive_embeddings, shared_features
            return logits, contrastive_embeddings

        if return_features:
            return logits, shared_features
        return logits

    def freeze_encoder(self, freeze: bool):
        """Freeze or unfreeze the shared encoder."""
        for param in self.shared_encoder.parameters():
            param.requires_grad = not freeze

        if freeze:
            self.shared_encoder.eval()
        else:
            self.shared_encoder.train()

    def freeze_contrastive_head(self, freeze: bool):
        """Freeze or unfreeze the contrastive head."""
        if not self.use_contrastive:
            raise ValueError(
                "SimplePredictor must have use_contrastive=True "
                "to support contrastive head freezing."
            )

        for param in self.contrastive_head.parameters():
            param.requires_grad = not freeze

        if freeze:
            self.contrastive_head.eval()
        else:
            self.contrastive_head.train()

    def freeze_classification_head(self, freeze: bool):
        """Freeze or unfreeze the classification head."""
        for param in self.classification_head.parameters():
            param.requires_grad = not freeze

        if freeze:
            self.classification_head.eval()
        else:
            self.classification_head.train()


__all__ = ["SimplePredictor"]
