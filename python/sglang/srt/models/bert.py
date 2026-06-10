# BERT模型实现文件 - 包含BERT嵌入层、编码器、池化器及序列分类模型的SGLang推理实现
# SPDX-License-Identifier: Apache-2.0
from typing import Iterable, Optional, Set, Tuple # 导入类型提示相关模块

import torch # 导入PyTorch深度学习框架
from torch import nn # 导入PyTorch神经网络模块

from sglang.srt.distributed import get_tensor_model_parallel_world_size # 导入获取张量并行世界大小的函数
from sglang.srt.layers.activation import get_act_fn # 导入获取激活函数的工具
from sglang.srt.layers.linear import ( # 导入并行线性层
    ColumnParallelLinear, # 列并行线性层
    QKVParallelLinear, # QKV并行线性层
    RowParallelLinear, # 行并行线性层
)
from sglang.srt.layers.pooler import CrossEncodingPooler, Pooler, PoolingType # 导入池化相关组件
from sglang.srt.layers.quantization.base_config import QuantizationConfig # 导入量化配置基类
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention # 导入注意力机制相关组件
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding # 导入词表并行嵌入层
from sglang.srt.model_executor.forward_batch_info import ForwardBatch # 导入前向批次信息类
from sglang.srt.model_loader.weight_utils import default_weight_loader # 导入默认权重加载器
from sglang.srt.server_args import get_global_server_args # 导入获取全局服务器参数的函数
from sglang.srt.utils import add_prefix # 导入添加前缀的工具函数

BertConfig = None # BERT配置类占位，将在运行时由transformers库填充


class BertEmbedding(nn.Module): # BERT嵌入层类，包含词嵌入、位置嵌入和token类型嵌入

    def __init__(self, config: BertConfig): # 初始化BERT嵌入层

        super().__init__() # 调用父类初始化
        self.size = config.hidden_size # 隐藏层大小
        self.word_embeddings = VocabParallelEmbedding( # 词嵌入层
            config.vocab_size, config.hidden_size # 词表大小和隐藏维度
        )
        self.position_embeddings = VocabParallelEmbedding( # 位置嵌入层
            config.max_position_embeddings, config.hidden_size # 最大位置数和隐藏维度
        )
        self.token_type_embeddings = VocabParallelEmbedding( # token类型嵌入层（用于区分句子对）
            config.type_vocab_size, config.hidden_size # 类型词表大小和隐藏维度
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # 层归一化
        self.position_ids = nn.Parameter( # 位置ID参数
            torch.empty((1, config.max_position_embeddings)), # 形状为(1, 最大位置数)
        )

        self.position_embedding_type = config.position_embedding_type # 位置嵌入类型
        if self.position_embedding_type != "absolute": # 仅支持绝对位置嵌入
            raise ValueError(
                "Only 'absolute' position_embedding_type" + " is supported" # 仅支持绝对位置嵌入类型
            )

    def forward( # 前向传播：计算输入token的嵌入表示
        self,
        input_ids: torch.Tensor, # 输入token ID张量
        positions: torch.Tensor, # 位置索引张量
        forward_batch: ForwardBatch, # 前向批次信息
    ) -> torch.Tensor:
        input_shape = input_ids.size() # 获取输入形状

        # Input embeddings.  # 输入词嵌入
        inputs_embeds = self.word_embeddings(input_ids) # 计算词嵌入

        # Position embeddings.  # 位置嵌入
        position_embeddings = self.position_embeddings(positions) # 计算位置嵌入

        token_type_ids = forward_batch.token_type_ids # 获取token类型ID

        if token_type_ids is None: # 如果没有提供token类型ID
            token_type_ids = torch.zeros( # 创建全零的token类型ID
                input_shape, dtype=torch.long, device=inputs_embeds.device # 与输入相同的形状和设备
            )

        token_type_embeddings = self.token_type_embeddings(token_type_ids) # 计算token类型嵌入

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings # 三种嵌入相加
        embeddings = self.LayerNorm(embeddings) # 层归一化
        return embeddings # 返回嵌入结果


class BertPooler(nn.Module): # BERT池化层类，取第一个token的隐藏状态并通过全连接层

    def __init__(self, config: BertConfig): # 初始化BERT池化层
        super().__init__() # 调用父类初始化
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # 全连接层
        self.activation = nn.Tanh() # Tanh激活函数

    def forward( # 前向传播：对隐藏状态进行池化
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch # 隐藏状态张量和前向批次信息
    ) -> torch.Tensor:
        # simply taking the hidden state corresponding  # 简单地取对应的隐藏状态
        first_token_tensor = hidden_states[0, :] # 取第一个token的隐藏状态

        pooled_output = self.dense(first_token_tensor) # 通过全连接层
        pooled_output = self.activation(pooled_output) # 通过激活函数

        return pooled_output # 返回池化输出


class BertEncoder(nn.Module): # BERT编码器类，包含多层BertLayer

    def __init__( # 初始化BERT编码器
        self,
        config: BertConfig, # BERT配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.config = config # 保存配置
        self.quant_config = quant_config # 保存量化配置
        self.layer = nn.ModuleList( # 创建多层BertLayer的模块列表
            [
                BertLayer(
                    config=config, # BERT配置
                    layer_id=layer_idx, # 层索引
                    quant_config=quant_config, # 量化配置
                    prefix=f"{prefix}.layer.{layer_idx}", # 参数名前缀
                )
                for layer_idx in range(config.num_hidden_layers) # 遍历所有隐藏层
            ]
        )

    def forward( # 前向传播：依次通过所有BertLayer
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch # 隐藏状态和前向批次信息
    ) -> torch.Tensor:
        for layer in self.layer: # 遍历每一层
            hidden_states = layer(hidden_states, forward_batch) # 通过当前层
        return hidden_states # 返回编码后的隐藏状态


class BertLayer(nn.Module): # BERT单层类，包含注意力层、中间层和输出层

    def __init__( # 初始化BERT单层
        self,
        config: BertConfig, # BERT配置
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化

        self.layer_id = layer_id # 保存层ID

        self.attention = BertAttention( # 自注意力层
            hidden_size=config.hidden_size, # 隐藏维度
            num_attention_heads=config.num_attention_heads, # 注意力头数
            layer_id=layer_id, # 层ID
            layer_norm_eps=config.layer_norm_eps, # 层归一化epsilon
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.attention", # 参数名前缀
        )

        self.intermediate = BertIntermediate( # 中间层（FFN第一部分）
            hidden_size=config.hidden_size, # 隐藏维度
            intermediate_size=config.intermediate_size, # 中间层维度
            hidden_act=config.hidden_act, # 隐藏层激活函数
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.intermediate", # 参数名前缀
        )

        self.output = BertOutput( # 输出层（FFN第二部分）
            hidden_size=config.hidden_size, # 隐藏维度
            intermediate_size=config.intermediate_size, # 中间层维度
            layer_norm_eps=config.layer_norm_eps, # 层归一化epsilon
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.output", # 参数名前缀
        )

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch): # 前向传播：注意力→中间层→输出
        attn_output = self.attention(hidden_states, forward_batch) # 通过注意力层
        intermediate_output = self.intermediate(attn_output) # 通过中间层
        output = self.output(intermediate_output, attn_output) # 通过输出层

        return output # 返回层输出


class BertAttention(nn.Module): # BERT注意力类，包含自注意力和输出投影

    def __init__( # 初始化BERT注意力层
        self,
        hidden_size: int, # 隐藏维度
        num_attention_heads: int, # 注意力头数
        layer_norm_eps: float, # 层归一化epsilon
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化

        self.self_attn = BertSelfAttention( # 自注意力层
            hidden_size=hidden_size, # 隐藏维度
            num_attention_heads=num_attention_heads, # 注意力头数
            layer_id=layer_id, # 层ID
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.output", # 参数名前缀
        )

        self.output = BertSelfOutput( # 自注意力输出层
            hidden_size=hidden_size, # 隐藏维度
            layer_norm_eps=layer_norm_eps, # 层归一化epsilon
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.output", # 参数名前缀
        )

    def forward( # 前向传播：计算自注意力并投影输出
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch # 隐藏状态和前向批次信息
    ) -> torch.Tensor:
        self_output = self.self_attn(hidden_states, forward_batch) # 通过自注意力层
        return self.output(self_output, hidden_states) # 通过输出层并返回


class BertSelfAttention(nn.Module): # BERT自注意力类，实现多头自注意力机制

    def __init__( # 初始化BERT自注意力层
        self,
        hidden_size: int, # 隐藏维度
        num_attention_heads: int, # 注意力头数
        layer_id: int = 0, # 层ID
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.hidden_size = hidden_size # 保存隐藏维度
        tp_size = get_tensor_model_parallel_world_size() # 获取张量并行大小

        self.total_num_heads = num_attention_heads # 总注意力头数
        assert self.total_num_heads % tp_size == 0 # 确保头数可被并行度整除

        self.num_heads = self.total_num_heads // tp_size # 每个并行分片的头数
        self.total_num_kv_heads = self.total_num_heads # KV头数与Q头数相同（非GQA）
        self.head_dim = self.hidden_size // self.total_num_heads # 每个头的维度
        assert self.head_dim * self.total_num_heads == self.hidden_size # 确保维度匹配

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size) # 每个并行分片的KV头数

        self.q_size = self.num_heads * self.head_dim # Q的总维度
        self.kv_size = self.num_kv_heads * self.head_dim # KV的总维度
        self.scaling = self.head_dim**-0.5 # 缩放因子
        self.qkv_proj = QKVParallelLinear( # QKV联合投影层
            hidden_size=self.hidden_size, # 输入维度
            head_size=self.head_dim, # 每个头的维度
            total_num_heads=self.total_num_heads, # 总Q头数
            total_num_kv_heads=self.total_num_kv_heads, # 总KV头数
            bias=True, # 使用偏置
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.qkv_proj", # 参数名前缀
        )

        self.attn = RadixAttention( # 基数注意力层
            num_heads=self.num_heads, # 注意力头数
            head_dim=self.head_dim, # 每个头的维度
            scaling=self.scaling, # 缩放因子
            num_kv_heads=self.num_kv_heads, # KV头数
            layer_id=layer_id, # 层ID
            prefix=f"{prefix}.attn", # 参数名前缀
            attn_type=AttentionType.ENCODER_ONLY, # 仅编码器类型注意力
        )

    def forward( # 前向传播：计算QKV并执行注意力
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch # 隐藏状态和前向批次信息
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states) # 计算QKV投影
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # 拆分QKV
        output = self.attn(q, k, v, forward_batch) # 执行注意力计算
        return output # 返回注意力输出


class BertSelfOutput(nn.Module): # BERT自注意力输出类，包含线性投影和层归一化

    def __init__( # 初始化自注意力输出层
        self,
        hidden_size: int, # 隐藏维度
        layer_norm_eps: float, # 层归一化epsilon
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.dense = RowParallelLinear( # 行并行线性层
            input_size=hidden_size, # 输入维度
            output_size=hidden_size, # 输出维度
            bias=True, # 使用偏置
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.dense", # 参数名前缀
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) # 层归一化

    def forward( # 前向传播：线性投影+残差连接+层归一化
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor # 隐藏状态和输入张量
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states) # 通过线性投影
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # 残差连接后层归一化
        return hidden_states # 返回输出


class BertIntermediate(nn.Module): # BERT中间层类，FFN的第一部分（升维+激活）

    def __init__( # 初始化BERT中间层
        self,
        hidden_size: int, # 隐藏维度
        intermediate_size: int, # 中间层维度
        hidden_act: str, # 激活函数名称
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.dense = ColumnParallelLinear( # 列并行线性层（升维）
            input_size=hidden_size, # 输入维度
            output_size=intermediate_size, # 输出维度（中间层维度）
            bias=True, # 使用偏置
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.dense", # 参数名前缀
        )
        self.intermediate_act_fn = get_act_fn(hidden_act) # 获取激活函数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # 前向传播：线性升维+激活
        hidden_states, _ = self.dense(hidden_states) # 通过线性层升维
        hidden_states = self.intermediate_act_fn(hidden_states) # 通过激活函数
        return hidden_states # 返回中间层输出


class BertOutput(nn.Module): # BERT输出层类，FFN的第二部分（降维+残差+层归一化）

    def __init__( # 初始化BERT输出层
        self,
        hidden_size: int, # 隐藏维度
        intermediate_size: int, # 中间层维度
        layer_norm_eps: float, # 层归一化epsilon
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化

        self.dense = RowParallelLinear( # 行并行线性层（降维）
            input_size=intermediate_size, # 输入维度（中间层维度）
            output_size=hidden_size, # 输出维度（隐藏维度）
            bias=True, # 使用偏置
            quant_config=quant_config, # 量化配置
            prefix=f"{prefix}.dense", # 参数名前缀
        )

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) # 层归一化

    def forward( # 前向传播：线性降维+残差连接+层归一化
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor # 隐藏状态和输入张量
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states) # 通过线性层降维
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # 残差连接后层归一化
        return hidden_states # 返回输出


class BertModel(nn.Module): # BERT模型主类，组合嵌入层、编码器和池化层

    def __init__( # 初始化BERT模型
        self,
        *,
        config: BertConfig, # BERT配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        use_bert_pooler: bool = False, # 是否使用BERT原生池化层
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化
        self.use_bert_pooler = use_bert_pooler # 是否使用BERT原生池化层
        self.config = config # 保存配置
        self.embeddings = BertEmbedding(config) # 嵌入层
        self.encoder = BertEncoder( # 编码器
            config=config, # BERT配置
            quant_config=quant_config, # 量化配置
            prefix=add_prefix("encoder", prefix), # 参数名前缀
        )
        pooling_type = ( # 池化类型
            PoolingType.CLS # CLS池化
            if get_global_server_args().is_embedding # 如果是嵌入模式
            else PoolingType.LAST # 否则使用最后一个token池化
        )
        self.pooler = ( # 池化层
            BertPooler(config) # 使用BERT原生池化层
            if self.use_bert_pooler # 如果指定使用
            else Pooler(pooling_type=pooling_type, normalize=True) # 否则使用通用池化层
        )

    @torch.no_grad() # 禁用梯度计算
    def forward( # 前向传播：嵌入→编码→池化
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置索引
        forward_batch: ForwardBatch, # 前向批次信息
        input_embeds: torch.Tensor = None, # 输入嵌入（可选）
        get_embedding: bool = False, # 是否获取嵌入表示
    ) -> torch.Tensor:
        assert get_embedding == True # 断言必须获取嵌入表示
        # Your tokenized IDs  # 你的分词ID

        hidden_states = self.embeddings( # 计算嵌入
            input_ids=input_ids, # 输入token ID
            positions=positions, # 位置索引
            forward_batch=forward_batch, # 前向批次信息
        )

        hidden_states = self.encoder(hidden_states, forward_batch=forward_batch) # 通过编码器

        if not self.use_bert_pooler: # 如果不使用BERT原生池化层
            hidden_states = self.pooler(hidden_states, forward_batch) # 通过池化层

        return hidden_states # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]: # 加载模型权重
        stacked_params_mapping = [ # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "query", "q"), # Q投影映射
            ("qkv_proj", "key", "k"), # K投影映射
            ("qkv_proj", "value", "v"), # V投影映射
        ]

        params_dict = dict(self.named_parameters()) # 获取参数字典
        for name, loaded_weight in weights: # 遍历所有权重
            name = name.replace("self", "self_attn") # 替换self为self_attn
            if not self.use_bert_pooler and "pooler" in name: # 如果不使用pooler且名称中包含pooler
                continue # 跳过
            for param_name, weight_name, shard_id in stacked_params_mapping: # 遍历堆叠参数映射

                if weight_name not in name: # 如果权重名不在参数名中
                    continue # 跳过
                name = name.replace(weight_name, param_name) # 替换权重名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict: # 如果是偏置且不在参数字典中
                    continue # 跳过
                param = params_dict[name] # 获取参数
                weight_loader = param.weight_loader # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id) # 加载权重分片
                break # 跳出内层循环
            else: # 如果没有匹配到堆叠参数
                # Skip loading extra bias for GPTQ models.  # 跳过GPTQ模型的额外偏置加载
                if name.endswith(".bias") and name not in params_dict: # 如果是偏置且不在参数字典中
                    continue # 跳过
                param = params_dict[name] # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
                weight_loader(param, loaded_weight) # 加载权重


class Contriever(BertModel): # Contriever模型类，继承自BertModel，用于信息检索
    pass # 无额外实现


class BertForSequenceClassification(nn.Module): # BERT序列分类模型类

    def __init__( # 初始化BERT序列分类模型
        self,
        *,
        config: BertConfig, # BERT配置
        quant_config: Optional[QuantizationConfig] = None, # 量化配置
        prefix: str = "", # 参数名前缀
    ):
        super().__init__() # 调用父类初始化

        self.num_labels = config.num_labels # 分类标签数
        self.bert = BertModel( # BERT模型
            config=config, # BERT配置
            quant_config=quant_config, # 量化配置
            use_bert_pooler=True, # 使用BERT原生池化层
            prefix=add_prefix("bert", prefix), # 参数名前缀
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) # 分类器线性层
        self.pooler = CrossEncodingPooler(config, self.classifier, self.bert.pooler) # 交叉编码池化层

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): # 加载模型权重
        self_weights = [] # 存储非bert部分的权重

        def weight_filter(): # 权重过滤器，分离bert权重和其他权重
            for name, weight in weights: # 遍历所有权重
                if name.startswith("bert."): # 如果权重属于bert模块
                    yield (name[len("bert.") :], weight) # 去掉bert前缀后yield
                else: # 否则
                    self_weights.append((name, weight)) # 添加到自身权重列表

        self.bert.load_weights(weight_filter()) # 加载bert权重

        params_dict = dict(self.named_parameters()) # 获取参数字典

        for name, loaded_weight in self_weights: # 遍历非bert权重
            if name.startswith("classifier"): # 如果是分类器权重
                param = params_dict[name] # 获取参数
                weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取权重加载器
                weight_loader(param, loaded_weight) # 加载权重

    def forward( # 前向传播：通过BERT模型和池化层得到分类结果
        self,
        input_ids: torch.Tensor, # 输入token ID
        positions: torch.Tensor, # 位置索引
        forward_batch: ForwardBatch, # 前向批次信息
        input_embeds: torch.Tensor = None, # 输入嵌入（可选）
        get_embedding: bool = False, # 是否获取嵌入表示
    ) -> torch.Tensor:
        assert get_embedding == True # 断言必须获取嵌入表示

        hidden_states = self.bert( # 通过BERT模型
            input_ids=input_ids, # 输入token ID
            positions=positions, # 位置索引
            forward_batch=forward_batch, # 前向批次信息
            input_embeds=input_embeds, # 输入嵌入
            get_embedding=get_embedding, # 获取嵌入标志
        )
        return self.pooler(hidden_states, forward_batch) # 通过池化层返回分类结果


EntryClass = [BertModel, Contriever, BertForSequenceClassification] # 入口类列表，用于模型注册
