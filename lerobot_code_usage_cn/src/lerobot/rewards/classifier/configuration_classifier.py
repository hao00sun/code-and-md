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

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode
from lerobot.configs.rewards import RewardModelConfig
from lerobot.optim import AdamWConfig, LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import OBS_IMAGE


@RewardModelConfig.register_subclass(name="reward_classifier")
@dataclass
class RewardClassifierConfig(RewardModelConfig):
    """Configuration for the Reward Classifier model."""

    name: str = "reward_classifier"

    # ----------------------------
    # Model structure
    # ----------------------------
    num_classes: int = 2
    hidden_dim: int = 256
    latent_dim: int = 256
    image_embedding_pooling_dim: int = 8
    dropout_rate: float = 0.1

    # Pretrained visual backbone.
    # 可以是 Hugging Face repo，例如 "lerobot/resnet10"
    # 也可以是本地路径，例如：
    # "/media/wu/.../hil-serl/pretrained_models/lerobot_resnet10"
    model_name: str = "lerobot/resnet10"

    device: str = "cpu"
    model_type: str = "cnn"  # "transformer" or "cnn"
    num_cameras: int = 2

    # ----------------------------
    # Optimization
    # ----------------------------
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    # ----------------------------
    # Label mapping
    # ----------------------------
    # 训练时从 batch 中哪个字段读取标签。
    #
    # 原始 LeRobot reward classifier 通常使用 reward / next.reward。
    #
    # 但你的 ARX 数据中 is_failure_data 没有进入 batch，
    # batch 中保留了 index，所以推荐使用：
    #
    #   label_key = "index"
    #
    # 然后通过 index 回查原始 parquet 中的 is_failure_data。
    label_key: str = "next.reward"

    # 是否做 1 - label 映射。
    #
    # 对你的数据：
    #   is_failure_data = 1 表示 failure
    #   is_failure_data = 0 表示 success / non-failure
    #
    # reward classifier 希望：
    #   label = 0 表示 failure
    #   label = 1 表示 success / non-failure
    #
    # 所以当 label_lookup_column = "is_failure_data" 时，
    # 需要 label_invert = True。
    label_invert: bool = False

    # 当 label_key = "index" 时使用。
    #
    # 指向原始 LeRobotDataset 根目录，例如：
    #
    # /media/wu/4750eeba-1063-264f-b8b7-b4e41ee3cc2f/datasets/arx5/arx_bimanual_0611_1511_v30
    #
    # reward classifier 会读取 root/data/**/*.parquet，
    # 建立：
    #
    #   global frame index -> is_failure_data
    #
    # 的查表。
    label_lookup_root: str | None = None

    # 从 parquet 中读取哪一列作为原始标签。
    #
    # 你的数据使用：
    #
    #   is_failure_data
    #
    label_lookup_column: str = "is_failure_data"

    # ----------------------------
    # Normalization
    # ----------------------------
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
        }
    )

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        """Validate feature configurations."""
        has_image = any(key.startswith(OBS_IMAGE) for key in self.input_features)
        if not has_image:
            raise ValueError(
                "You must provide an image observation "
                "(key starting with 'observation.image') in the input features"
            )