# custom_llama.py

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig

class GridEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.origin_embedding = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        if config.scaled_position_embedding:
            self.row_embedding = nn.Embedding(1, config.hidden_size)
            self.col_embedding = nn.Embedding(1, config.hidden_size)
        else:
            self.row_embedding = nn.Embedding(config.max_rows, config.hidden_size)
            self.col_embedding = nn.Embedding(config.max_cols, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _grid_embed(self, grid_size):
        # example data
        # /n2 2 2/n2 2 2/n2 2 2
        if config.scaled_position_embeddings:
            pos_row_embed = self.pos_row_embed(torch.zeros(config.max_rows, dtype=torch.long, device=device))
            pos_col_embed = self.pos_col_embed(torch.zeros(config.max_cols, dtype=torch.long, device=device))
            pos_row_embeds = (torch.arange(1, config.max_rows + 1, device=device)[:, None] * pos_row_embed)
            pos_col_embeds = (torch.arange(1, config.max_cols + 1, device=device)[:, None] * pos_col_embed)
            pos_embed = pos_row_embeds[:, None, None, :] + pos_col_embeds[None, :, None, :]
        else:
            pos_row_embed = self.pos_row_embed(torch.arange(config.max_rows, dtype=torch.long, device=device))
            pos_col_embed = self.pos_col_embed(torch.arange(config.max_cols, dtype=torch.long, device=device))
            pos_embed = pos_row_embed[:, None, None, :] + pos_col_embed[None, :, None, :]

    def grid_embedding(self, input_ids, prefix_counts):
        # prefix counts = encode grid counts
        return 0

    def forward(self, input_ids):
        
        embed_tokens = self.origin_embedding(input_ids) + self.grid_embedding

        return embed_tokens
   
 # encoder input
    # # input_ids=encoder_input_ids,
    #     attention_mask=encoder_attention_mask,
    #     labels=encoder_labels,
    #     output_hidden_states=True,
    #     prefix_counts = prefix_counts,     
class CustomLlamaModel(LlamaModel):
    def __init__(self, config, extra_size):
        super().__init__(config)
        self.set_input_embeddings = GridEmbedding(config.vocab_size, config.hidden_size, extra_size)

    def forward(self, input_ids, labels, prefix_counts, attention_mask=None, output_hidden_states=True):
       
        inputs_embeds = self.embed_tokens(input_ids, prefix_counts)  # 传递额外参数
        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, output_hidden_states=True)

# 先加载 encoder 和 decoder 的 tokenizer（这里以 Llama 模型为例）
encoder_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
decoder_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# 实例化数据集，参数仅为示例，实际使用时请根据需要调整
train_dataset = TrainDataset(
    train_data_dir="path/to/train/data",         # 训练数据所在目录
    eval_train_dir="path/to/eval_train/data",      # 评估数据所在目录
    verifier_file="path/to/verifiers.py",          # verifier 脚本路径
    encoder_tokenizer=encoder_tokenizer,
    decoder_tokenizer=decoder_tokenizer,
    total_steps=1000,
    min_prefix=2,
    max_prefix=7,
    max_seq_len=512,
    augment_ratio=0.0,
    augment_single_grid=False,
    seed=42,
    process_index=0,
    ntokens=64,
    debug_fixed_train_order=False,
    debug_random_pad=False,
    debug_pad_len=-1,
    encoder_pad_side="right",
    decoder_pad_side="right",
    encoder_loss_type="rest",
    anti_invar_ratio=0.0,
    num_workers=0,
)

# 构造 DataLoader，使用 collate_fn_train 整理 batch 数据
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn_train,
    drop_last=True,
    num_workers=0,
)

# 查看 DataLoader 输出的格式
for batch in train_loader:
    print(batch)
    break