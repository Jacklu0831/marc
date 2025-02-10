# custom_llama.py

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig
from torch.utils.data import DataLoader
from data_utils import TrainDataset, collate_fn_train
from transformers import AutoTokenizer

from accelerate import Accelerator
from functools import partial




class GridEmbedding(nn.Module):

    def __init__(self, config,tokenizer,scaled_position_embedding=True):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.origin_embedding = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.scaled_position_embedding = scaled_position_embedding
        if self.scaled_position_embedding:
            self.row_embedding = nn.Embedding(1, config.hidden_size)
            self.col_embedding = nn.Embedding(1, config.hidden_size)
        else:
            self.row_embedding = nn.Embedding(config.max_rows, config.hidden_size)
            self.col_embedding = nn.Embedding(config.max_cols, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _grid_embed(self, width,height):
    
        if self.scaled_position_embeddings:
            pos_row_embed = self.row_embedding(torch.zeros(height, dtype=torch.long, device=device))
            pos_col_embed = self.col_embedding(torch.zeros(width, dtype=torch.long, device=device))
            pos_row_embeds = (torch.arange(1, height + 1, device=device)[:, None] * pos_row_embed).unsqueeze(1)
            pos_col_embeds = (torch.arange(1, width + 1, device=device)[:, None] * pos_col_embed).unsqueeze(1)
            pos_embed = pos_row_embeds[:, None, :] + pos_col_embeds[None, :, :]
        else:
            pos_row_embed = self.row_embedding(torch.arange(config.max_rows, dtype=torch.long, device=device))
            pos_col_embed = self.col_embedding(torch.arange(config.max_cols, dtype=torch.long, device=device))
            pos_embed = pos_row_embed[:, None, :]  + pos_col_embed[None, :, :]

        return pos_embed


    def _get_grid_info(self, tokens, input_token_positions, output_token_positions):
        """
        get the start ptr of each grid, 
        get the height and width of each grid
        """
        grid_starts_ptrs = []
        heights = []
        widths = []
        for i in input_token_positions:
            count = 0
            idx = i
            height = []
            width = []
            while count < 1:
                if tokens[idx+2]==13: 
                    #13 = '\n'
                    break
                else:
                    height.append(token[idx+2])
                    idx += 1
            height = height.join('')
            while count < 2:
                if tokens[idx+2]==13: 
                    #13 = '\n'
                    break
                else:
                    width.append(token[idx+2])
                    idx += 1
            width = width.join('')
            heights.append(int(height))
            widths.append(int(width))
            grid_starts_ptrs.append(idx+1) 
        
        return grid_starts_ptrs, heights, widths
    
    

    def grid_embedding(self, input_ids, input_token_positions, output_token_positions):

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        grid_embeds = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device)

        for b in range(batch_size):
            sample_ids = input_ids[b]
            tokens = self.tokenizer.convert_ids_to_tokens(sample_ids.tolist())
            grid_starts_ptrs, heights, widths = self._get_grid_info(tokens, input_token_positions, output_token_positions)
            
            for p, idx in enumerate(grid_starts_ptrs):
                grid_pos_embed = self._grid_embed(widths[idx],heights[idx])
                for i in range(heights[idx]):
                    grid_embeds[p+i*(widths[idx]+1):p+i*(widths[idx]+1)+widths[idx]] = grid_pos_embed[i,:]
            
       
        
        grid_embeds = self.dropout(grid_embeds)
        return grid_embeds

      


    def forward(self, input_ids, input_token_positions, output_token_positions):
        
        embed_tokens = self.origin_embedding(input_ids) + self.grid_embedding(input_ids, input_token_positions, output_token_positions)

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

    def forward(self, input_ids, labels, input_token_positions, output_token_positions,attention_mask=None, output_hidden_states=True):
       
        inputs_embeds = self.embed_tokens(input_ids, input_token_positions, output_token_positions )  # 传递额外参数
        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, output_hidden_states=True)










import torch

def validate_grid_embedding(grid_embedding_fn, input_ids, prefix_counts, input_token_positions, output_token_positions):
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    print("===== Grid Embedding Validation =====")
    
    for batch_idx in range(batch_size):
        print(f"\nBatch {batch_idx}:")
        input_positions = input_token_positions[batch_idx]
        output_positions = output_token_positions[batch_idx]

        # 获取 input grid 的信息
        num_input_tokens = prefix_counts[batch_idx][0]
        height = input_ids[batch_idx, num_input_tokens].item()
        width = input_ids[batch_idx, num_input_tokens + 1].item()
        start_idx = num_input_tokens + 2

        row_index = (input_positions - start_idx) // width
        col_index = (input_positions - start_idx) % width

        print(f"  Input Grid: height={height}, width={width}")
        print(f"  Input Positions: {input_positions.tolist()}")
        print(f"  Computed Row Index: {row_index.tolist()}")
        print(f"  Computed Col Index: {col_index.tolist()}")

        # 获取 output grid 的信息
        num_output_tokens = prefix_counts[batch_idx][1]
        height_out = input_ids[batch_idx, num_output_tokens].item()
        width_out = input_ids[batch_idx, num_output_tokens + 1].item()
        start_idx_out = num_output_tokens + 2

        row_index_out = (output_positions - start_idx_out) // width_out
        col_index_out = (output_positions - start_idx_out) % width_out

        print(f"  Output Grid: height={height_out}, width={width_out}")
        print(f"  Output Positions: {output_positions.tolist()}")
        print(f"  Computed Row Index (Output): {row_index_out.tolist()}")
        print(f"  Computed Col Index (Output): {col_index_out.tolist()}")

    print("===== Validation Complete =====\n")

# 伪造测试数据
batch_size = 2
seq_len = 30
device = torch.device("cpu")

# 假设的 input_ids（用索引填充模拟）
input_ids = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.long, device=device)

# 假设 prefix_counts
prefix_counts = torch.tensor([[3, 15], [4, 20]], dtype=torch.long, device=device)

# 假设 input 和 output 的 token 位置
input_token_positions = torch.tensor([[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], dtype=torch.long, device=device)
output_token_positions = torch.tensor([[20, 21, 22], [25, 26, 27]], dtype=torch.long, device=device)

# 运行验证
validate_grid_embedding(None, input_ids, prefix_counts, input_token_positions, output_token_positions)
