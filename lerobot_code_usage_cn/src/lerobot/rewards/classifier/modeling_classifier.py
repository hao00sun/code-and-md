# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.utils.constants import OBS_IMAGE, REWARD

from ..pretrained import PreTrainedRewardModel
from .configuration_classifier import RewardClassifierConfig


class ClassifierOutput:
    """Wrapper for classifier outputs with additional metadata."""

    def __init__(
        self,
        logits: Tensor,
        probabilities: Tensor | None = None,
        hidden_states: Tensor | None = None,
    ):
        self.logits = logits
        self.probabilities = probabilities
        self.hidden_states = hidden_states

    def __repr__(self):
        return (
            f"ClassifierOutput(logits={self.logits}, "
            f"probabilities={self.probabilities}, "
            f"hidden_states={self.hidden_states})"
        )


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings

        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding.

        Supports arbitrary CNN feature-map sizes by adaptive-pooling
        them to the learned kernel spatial size.
        """

        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state

        added_batch_dim = False

        # Normalize feature layout to [B, C, H, W]
        if features.dim() == 3:
            added_batch_dim = True

            if features.shape[0] == self.channel:
                # [C, H, W] -> [1, C, H, W]
                features = features.unsqueeze(0)
            elif features.shape[-1] == self.channel:
                # [H, W, C] -> [1, C, H, W]
                features = features.permute(2, 0, 1).unsqueeze(0)
            else:
                raise RuntimeError(
                    f"Unexpected 3D feature shape: {features.shape}, expected channel={self.channel}"
                )

        elif features.dim() == 4:
            if features.shape[1] == self.channel:
                # already [B, C, H, W]
                pass
            elif features.shape[-1] == self.channel:
                # [B, H, W, C] -> [B, C, H, W]
                features = features.permute(0, 3, 1, 2)
            else:
                raise RuntimeError(
                    f"Unexpected 4D feature shape: {features.shape}, expected channel={self.channel}"
                )

        else:
            raise RuntimeError(f"Unexpected feature ndim: {features.ndim}, shape={features.shape}")

        # Example: [B, C, 15, 27] -> [B, C, 4, 4]
        if features.shape[-2:] != (self.height, self.width):
            features = F.adaptive_avg_pool2d(features, output_size=(self.height, self.width))

        # features: [B, C, H, W]
        # kernel:   [C, H, W, F]
        # output:   [B, C, F] -> [B, C*F]
        output = (features.unsqueeze(-1) * self.kernel.unsqueeze(0)).sum(dim=(2, 3))
        output = output.flatten(start_dim=1)

        if added_batch_dim:
            output = output.squeeze(0)

        return output


class Classifier(PreTrainedRewardModel):
    """Image classifier built on top of a pre-trained encoder."""

    name = "reward_classifier"
    config_class = RewardClassifierConfig

    def __init__(
        self,
        config: RewardClassifierConfig,
        **kwargs,
    ):
        from transformers import AutoModel

        super().__init__(config)
        self.config = config

        # Set up encoder
        encoder = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True)
        # Extract vision model if we're given a multimodal model
        if hasattr(encoder, "vision_model"):
            logging.info("Multimodal model detected - using vision encoder only")
            self.encoder = encoder.vision_model
            self.vision_config = encoder.config.vision_config
        else:
            self.encoder = encoder
            self.vision_config = getattr(encoder, "config", None)

        # Model type from config
        self.is_cnn = self.config.model_type == "cnn"

        # For CNNs, initialize backbone
        if self.is_cnn:
            self._setup_cnn_backbone()

        self._freeze_encoder()

        # Extract image keys from input_features
        self.image_keys = [
            key.replace(".", "_") for key in config.input_features if key.startswith(OBS_IMAGE)
        ]

        if self.is_cnn:
            self.encoders = nn.ModuleDict()
            for image_key in self.image_keys:
                encoder = self._create_single_encoder()
                self.encoders[image_key] = encoder

        self._build_classifier_head()

        self.label_lookup = None
        if getattr(self.config, "label_key", None) == "index":
            self._setup_label_lookup()

    def _setup_cnn_backbone(self):
        """Set up CNN encoder"""
        if hasattr(self.encoder, "fc"):
            self.feature_dim = self.encoder.fc.in_features
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        elif hasattr(self.encoder.config, "hidden_sizes"):
            self.feature_dim = self.encoder.config.hidden_sizes[-1]  # Last channel dimension
        else:
            raise ValueError("Unsupported CNN architecture")

    def _freeze_encoder(self) -> None:
        """Freeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _create_single_encoder(self):
        encoder = nn.Sequential(
            self.encoder,
            SpatialLearnedEmbeddings(
                height=4,
                width=4,
                channel=self.feature_dim,
                num_features=self.config.image_embedding_pooling_dim,
            ),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.feature_dim * self.config.image_embedding_pooling_dim, self.config.latent_dim),
            nn.LayerNorm(self.config.latent_dim),
            nn.Tanh(),
        )

        return encoder

    def _build_classifier_head(self) -> None:
        """Initialize the classifier head architecture."""
        # Get input dimension based on model type
        if self.is_cnn:
            input_dim = self.config.latent_dim
        else:  # Transformer models
            if hasattr(self.encoder.config, "hidden_size"):
                input_dim = self.encoder.config.hidden_size
            else:
                raise ValueError("Unsupported transformer architecture since hidden_size is not found")

        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim * self.config.num_cameras, self.config.hidden_dim),
            nn.Dropout(self.config.dropout_rate),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.hidden_dim,
                1 if self.config.num_classes == 2 else self.config.num_classes,
            ),
        )

    def _get_encoder_output(self, x: torch.Tensor, image_key: str) -> torch.Tensor:
        """Extract the appropriate output from the encoder."""
        with torch.no_grad():
            if self.is_cnn:
                # The HF ResNet applies pooling internally
                outputs = self.encoders[image_key](x)
                return outputs
            else:  # Transformer models
                outputs = self.encoder(x)
                return outputs.last_hidden_state[:, 0, :]

    def _setup_label_lookup(self) -> None:
        """Build global frame index -> label lookup from dataset parquet files.

        Used when the official dataloader drops custom columns such as
        is_failure_data but still keeps batch["index"].
        """
        root = getattr(self.config, "label_lookup_root", None)
        column = getattr(self.config, "label_lookup_column", "is_failure_data")

        if root is None:
            raise ValueError(
                "label_key='index' requires config.label_lookup_root to point to the LeRobot dataset root."
            )

        root = Path(root)
        data_dir = root / "data"

        if not data_dir.exists():
            raise FileNotFoundError(f"Cannot find data directory: {data_dir}")

        parquet_files = sorted(data_dir.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under: {data_dir}")

        index_to_label = {}

        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)

            if "index" not in df.columns:
                raise KeyError(f"{parquet_file} does not contain column 'index'")

            if column not in df.columns:
                raise KeyError(f"{parquet_file} does not contain column '{column}'")

            for idx, value in zip(df["index"].tolist(), df[column].tolist()):
                index_to_label[int(idx)] = int(value)

        if not index_to_label:
            raise RuntimeError(f"No labels loaded from {data_dir}")

        max_index = max(index_to_label.keys())
        lookup = torch.zeros(max_index + 1, dtype=torch.float32)

        for idx, value in index_to_label.items():
            lookup[idx] = float(value)

        # If label_lookup was previously created as a normal attribute,
        # remove it before registering it as a PyTorch buffer.
        if hasattr(self, "label_lookup"):
            delattr(self, "label_lookup")

        self.register_buffer("label_lookup", lookup, persistent=False)

        unique_values = torch.unique(lookup).detach().cpu().tolist()
        logging.info(
            "Loaded label lookup from %s, column=%s, num_labels=%d, max_index=%d, unique_values=%s",
            root,
            column,
            len(index_to_label),
            max_index,
            unique_values,
        )

    def extract_images_and_labels(self, batch: dict[str, Tensor]) -> tuple[list, Tensor]:
        """Extract image tensors and label tensors from batch.

        For ARX:
            label_key = "index"
            label_lookup_column = "is_failure_data"
            label_invert = True

        Mapping:
            is_failure_data=1 -> label=0 -> failure
            is_failure_data=0 -> label=1 -> success/non-failure
        """
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]

        label_key = getattr(self.config, "label_key", REWARD)
        label_invert = getattr(self.config, "label_invert", False)

        # IMPORTANT:
        # label_key == "index" must be handled before "label_key in batch",
        # otherwise batch["index"] will be wrongly used as the label.
        if label_key == "index":
            if self.label_lookup is None:
                raise RuntimeError("label_lookup is not initialized, but label_key='index'.")

            if "index" not in batch:
                raise KeyError(
                    f"label_key='index' requires batch['index'], but available keys are: {list(batch.keys())}"
                )

            indices = batch["index"].long().to(self.label_lookup.device)

            if indices.ndim > 1:
                indices = indices.squeeze(-1)

            labels = self.label_lookup[indices]

        elif label_key in batch:
            labels = batch[label_key]

        else:
            raise KeyError(
                f"Reward classifier expected label_key='{label_key}' in batch, "
                f"but available keys are: {list(batch.keys())}"
            )

        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)

        labels = labels.to(next(self.parameters()).device)

        if labels.ndim > 1:
            labels = labels.squeeze(-1)

        if label_invert:
            labels = 1 - labels

        return images, labels

    def predict(self, xs: list) -> ClassifierOutput:
        """Forward pass of the classifier for inference."""
        encoder_outputs = torch.hstack(
            [self._get_encoder_output(x, img_key) for x, img_key in zip(xs, self.image_keys, strict=True)]
        )
        logits = self.classifier_head(encoder_outputs)

        if self.config.num_classes == 2:
            logits = logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)

        return ClassifierOutput(logits=logits, probabilities=probabilities, hidden_states=encoder_outputs)

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        """Returns 1.0 for success, 0.0 for failure based on image observations."""
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]
        output = self.predict(images)

        if self.config.num_classes == 2:
            return (output.probabilities > 0.5).float()
        else:
            return torch.argmax(output.probabilities, dim=1).float()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Standard forward pass for training compatible with train.py."""
        # Extract images and labels
        images, labels = self.extract_images_and_labels(batch)

        # Get predictions
        outputs = self.predict(images)

        # Calculate loss
        if self.config.num_classes == 2:
            # Binary classification
            labels = labels.to(outputs.logits.device).float()

            if labels.ndim > 1:
                labels = labels.squeeze(-1)

            label_min = labels.min().item()
            label_max = labels.max().item()
            label_unique = torch.unique(labels.detach().cpu())

            if label_min < 0.0 or label_max > 1.0:
                raise RuntimeError(
                    f"Invalid binary labels. Expected labels in [0, 1], "
                    f"but got min={label_min}, max={label_max}, "
                    f"unique={label_unique}. "
                    f"Check label_key / label_lookup / label_invert."
                )

            loss = nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)
            predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        else:
            # Multi-class classification
            labels = labels.to(outputs.logits.device).long()

            if labels.ndim > 1:
                labels = labels.squeeze(-1)

            loss = nn.functional.cross_entropy(outputs.logits, labels)
            predictions = torch.argmax(outputs.logits, dim=1)

        # Calculate accuracy for logging
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        # Return loss and metrics for logging
        output_dict = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        return loss, output_dict

    def predict_reward(self, batch, threshold=0.5):
        """Eval method. Returns predicted reward with the decision threshold as argument."""
        # Extract images from batch dict
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]

        if self.config.num_classes == 2:
            probs = self.predict(images).probabilities
            logging.debug(f"Predicted reward images: {probs}")
            return (probs > threshold).float()
        else:
            return torch.argmax(self.predict(images).probabilities, dim=1)
