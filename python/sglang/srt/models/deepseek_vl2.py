# 本文件实现了 DeepSeek-VL2 视觉语言模型的推理代码，包括 MLP 投影器和因果语言模型。
# DeepSeek-VL2 是 DeepSeek 推出的多模态语言模型，使用 SigLIP 视觉编码器和
# 下采样 MLP 投影器将视觉特征映射到语言模型空间，支持全局-局部视图融合。

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from sglang.srt.configs.deepseekvl2 import (
    DeepseekVL2Config,
    DeepseekVL2MlpProjectorConfig,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek import DeepseekForCausalLM
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM


class DeepseekVL2MlpProjector(nn.Module):
    """DeepSeek-VL2 的 MLP 投影器，将视觉编码器的输出映射到语言模型的隐藏空间，
    支持 identity、linear、mlp_gelu 和 downsample_mlp_gelu 四种投影类型"""

    def __init__(
        self,
        config: DeepseekVL2MlpProjectorConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):

        super().__init__()

        self.config = config

        if config.projector_type == "identity":
            # 恒等映射，不进行任何变换
            modules = nn.Identity()

        elif config.projector_type == "linear":
            # 单层线性投影
            self.layers = nn.ModuleList(
                [
                    ReplicatedLinear(
                        config.input_dim,
                        config.n_embed,
                        quant_config=quant_config,
                    )
                ]
            )

        elif config.projector_type == "mlp_gelu":
            # 多层 MLP 投影，使用 GELU 激活
            mlp_depth = config.depth
            self.layers = nn.ModuleList(
                [
                    ReplicatedLinear(
                        config.input_dim,
                        config.n_embed,
                        quant_config=quant_config,
                    )
                ]
            )
            for _ in range(1, mlp_depth):
                self.layers.append(nn.GELU())
                self.layers.append(
                    ReplicatedLinear(
                        config.n_embed,
                        config.n_embed,
                        quant_config=quant_config,
                    )
                )

        elif config.projector_type == "downsample_mlp_gelu":
            # 下采样 MLP 投影，先下采样空间分辨率再通过 MLP 映射
            mlp_depth = config.depth
            mlp_ratio = config.mlp_ratio
            self.layers = nn.ModuleList(
                [
                    ReplicatedLinear(
                        config.input_dim
                        * config.downsample_ratio
                        * config.downsample_ratio,  # 下采样后通道维度增大
                        config.n_embed * mlp_ratio,
                        quant_config=quant_config,
                    )
                ]
            )
            for _ in range(1, mlp_depth - 1):
                self.layers.append(nn.GELU())
                self.layers.append(
                    ReplicatedLinear(
                        config.n_embed * mlp_ratio,
                        config.n_embed * mlp_ratio,
                        quant_config=quant_config,
                    )
                )
            self.layers.append(nn.GELU())
            self.layers.append(
                ReplicatedLinear(
                    config.n_embed * mlp_ratio,
                    config.n_embed,
                    quant_config=quant_config,
                )
            )

        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

        if config.token_pooling:
            # token 池化层，将 2x2 的 token 合并为一个
            self.token_pooling_layer = ReplicatedLinear(
                config.input_dim * 4, config.input_dim, quant_config=quant_config
            )

    def forward(self, x):
        if self.config.token_pooling:
            # token 池化：将 2x2 的 patch 合并为一个 token
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)  # 转为 [B, C, H, W]

            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # 2x2 滑动窗口
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            patches = patches.contiguous().view(
                batch_size, channels, h_patches * w_patches, -1
            )
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)[0]  # 线性投影合并后的 token

        elif self.config.projector_type == "downsample_mlp_gelu":
            # 下采样：通过 unfold 将空间分辨率降低，通道维度增大
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.config.downsample_ratio:
                pad = self.config.downsample_ratio - h % self.config.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)  # 填充使尺寸整除

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(
                x,
                kernel_size=self.config.downsample_ratio,
                stride=self.config.downsample_ratio,
                padding=0,
            )  # B, C*4, HW // 4  # 展开为下采样后的 token
            x = x.permute(0, 2, 1)

        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]  # ReplicatedLinear 返回元组，取第一个元素
        return x


class DeepseekVL2ForCausalLM(nn.Module):
    """DeepSeek-VL2 因果语言模型，结合视觉编码器（SigLIP ViT）、MLP 投影器和语言模型，
    支持全局-局部视图融合的多模态图像理解"""

    def __init__(
        self,
        config: DeepseekVL2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        # ----------- vision encoder ------------
        vision_config = config.vision_config
        self.vision = self._init_vision_module(vision_config, quant_config)

        # ----------- vl projector ------------
        projector_config = config.projector_config
        self.projector = DeepseekVL2MlpProjector(projector_config, quant_config)

        self.tile_tag = config.tile_tag  # 切片标签，当前仅支持 "2D"
        self.global_view_pos = config.global_view_pos  # 全局视图位置："head" 或 "tail"

        embed_std = 1 / torch.sqrt(
            torch.tensor(projector_config.n_embed, dtype=torch.float32)
        )
        if self.tile_tag == "2D":
            # 2D 切片模式下的换行符和视图分隔符
            self.image_newline = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
            self.view_seperator = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
        else:
            raise ValueError(f"tile tag should be 2D, but got {self.tile_tag}")

        # ----------- language model ------------
        language_config = config.language_config
        if language_config.use_mla:
            # 使用 MLA（Multi-head Latent Attention）的语言模型
            self.language_model = DeepseekV2ForCausalLM(language_config)
        else:
            # deepseek-vl2-tiny forbids mla
            # deepseek-vl2-tiny 禁用 MLA，使用标准注意力
            self.language_model = DeepseekForCausalLM(language_config)

    def _init_vision_module(
        self, vision_config, quant_config: Optional[QuantizationConfig]
    ) -> nn.Module:
        """初始化视觉编码器模块，使用 timm 库创建 SigLIP ViT 模型"""
        # TODO: refactor vision model through timm wrapper from transformers
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm") from ImportError

        model = timm.create_model(
            "vit_so400m_patch14_siglip_384.webli",
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )

        model = model.to(dtype=torch.get_default_dtype())
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ):
        """前向传播：通过通用多模态嵌入流程处理视觉和语言输入"""
        hs = general_mm_embed_routine(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            multimodal_model=self,
            language_model=self.language_model,
        )

        return hs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，语言模型权重委托给子模型加载，其余权重直接加载"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters())
        weights = list(weights)
        for name, loaded_weight in weights:
            if "language" in name:
                # 语言模型权重委托给子模型加载
                name = name.replace("language.", "")
                self.language_model.load_weights([(name, loaded_weight)])
            else:
                # 视觉编码器和投影器权重直接加载
                param = params_dict[name]
                weights_loader = getattr(param, "weight_loader", default_weight_loader)
                weights_loader(param, loaded_weight)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """对输入 token ID 进行填充，以适配多模态输入"""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]):
        """提取图像特征：通过视觉编码器和投影器处理图像，并融合全局和局部视图"""

        images_spatial_crop = torch.cat(
            [item.images_spatial_crop for item in items], dim=0
        )

        assert images_spatial_crop.dim() == 3

        # TODO: can it be batched ?
        images_in_this_batch = []
        for item in items:
            assert item.feature.dim() == 4
            # 通过视觉编码器提取特征
            image_feature = self.vision.forward_features(
                item.feature.type(next(self.vision.parameters()).dtype)
            )
            # 通过投影器映射到语言模型维度
            images_embeds = self.projector(image_feature)
            _, hw, n_dim = images_embeds.shape
            h = w = int(hw**0.5)
            tile_index = 0
            for jdx in range(item.images_spatial_crop.shape[1]):
                num_width_tiles, num_height_tiles = item.images_spatial_crop[0, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                num_tiles_in_image = num_width_tiles * num_height_tiles

                # [hw, D]
                global_features = images_embeds[tile_index]  # 全局视图特征

                # [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[
                    tile_index + 1 : tile_index + 1 + num_tiles_in_image
                ]  # 局部视图特征
                tile_index += num_tiles_in_image + 1

                # format global and local features
                # ----------------- global view add newline -----------------
                # [hw, D] -> [h, w, D]
                global_features = global_features.view(h, w, n_dim)

                # [D]     -> [h, 1, D]
                new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)

                # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                # 在全局视图每行末尾添加换行符
                global_features = torch.cat(
                    [global_features, new_lines_in_global], dim=1
                )

                # [h, w + 1, D] -> [h * (w + 1), D]
                global_features = global_features.view(-1, n_dim)

                # ----------------- local view add newline -----------------
                # [num_height_tiles * num_width_tiles, h * w, D] ->
                # [num_height_tiles * h, num_width_tiles * w, D]
                # 将局部视图重新排列为二维网格
                local_features = rearrange(
                    local_features,
                    "(th tw) (h w) d -> (th h) (tw w) d",
                    th=num_height_tiles,
                    tw=num_width_tiles,
                    h=h,
                    w=w,
                )

                # [D] -> [num_height_tiles * h, 1, D]
                new_lines_in_local = repeat(
                    self.image_newline,
                    "d -> (th h) 1 d",
                    th=num_height_tiles,
                    h=h,
                )

                # [num_height_tiles * h, num_width_tiles * w + 1, D]
                # 在局部视图每行末尾添加换行符
                local_features = torch.cat([local_features, new_lines_in_local], dim=1)

                # [num_height_tiles * h, num_width_tiles * w + 1, D]
                #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                local_features = local_features.view(-1, n_dim)

                # merge global and local tiles
                # 合并全局和局部视图，中间用视图分隔符分隔
                if self.global_view_pos == "head":
                    global_local_features = torch.cat(
                        [
                            global_features,
                            self.view_seperator[None, :],
                            local_features,
                        ]
                    )
                else:
                    global_local_features = torch.cat(
                        [
                            local_features,
                            self.view_seperator[None, :],
                            global_features,
                        ]
                    )

                images_in_this_batch.append(global_local_features)

        return torch.cat(images_in_this_batch, dim=0)


EntryClass = DeepseekVL2ForCausalLM
