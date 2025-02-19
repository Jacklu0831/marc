# custom_llama.py

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig, LlamaForCausalLM
from torch.utils.data import DataLoader
from data_utils import TrainDataset, collate_fn_train
from transformers import AutoTokenizer

from accelerate import Accelerator
from functools import partial



class GridEmbedding(nn.Embedding):

    def __init__(self, config, if_dropout=False, tokenizer=None, scaled_position_embedding=True):
        super().__init__(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.config = config
        self.tokenizer = tokenizer
        self.scaled_position_embedding = scaled_position_embedding

        if self.scaled_position_embedding:
            self.row_embedding = nn.Embedding(1, config.hidden_size)
            self.col_embedding = nn.Embedding(1, config.hidden_size)
        else:
            self.row_embedding = nn.Embedding(config.max_rows, config.hidden_size)
            self.col_embedding = nn.Embedding(config.max_cols, config.hidden_size)

        self.if_dropout = if_dropout
        if self.if_dropout:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _grid_embed(self, width, height, device):
        """Computes the grid positional embeddings."""
        # print(width)
        # print(height)
        if self.scaled_position_embedding:
            pos_row_embed = self.row_embedding(torch.zeros(height, dtype=torch.long, device=device))
            pos_col_embed = self.col_embedding(torch.zeros(width, dtype=torch.long, device=device))
            pos_row_embeds = (torch.arange(1, height + 1, device=device)[:, None] * pos_row_embed)
            pos_col_embeds = (torch.arange(1, width + 1, device=device)[:, None] * pos_col_embed)
            pos_embed = pos_row_embeds[:, None, :] + pos_col_embeds[None, :, :]
        else:
            pos_row_embed = self.row_embedding(torch.arange(height, dtype=torch.long, device=device))
            pos_col_embed = self.col_embedding(torch.arange(width, dtype=torch.long, device=device))
            pos_embed = pos_row_embed[:, None, :] + pos_col_embed[None, :, :]

        return pos_embed

    def _get_grid_info(self, tokens, input_token_positions):
        """Extracts grid metadata from token sequences."""
        grid_starts_ptrs, heights, widths = [], [], []
        # for debug
        # print(type(input_token_positions))
        # print(input_token_positions)
        for i in input_token_positions:
            # print(type(i))
            # print(i)
            idx = i
            height, width = [], []

            while idx + 2 < len(tokens) and tokens[idx + 2] != 12:  # '\n'
                height.append(str(tokens[idx + 2].item()))  # Convert token ID to char
                idx += 1
            # print(f"height:{height}")
            idx += 1  # Skip '\n'
            while idx + 2 < len(tokens) and tokens[idx + 2] != 12:  # '\n'
                width.append(str(tokens[idx + 2].item()))
                idx += 1
            # print(f"width:{width}")

            heights.append(int("".join(height)) if height else 0)
            widths.append(int("".join(width)) if width else 0)
            grid_starts_ptrs.append(idx + 1)

        return grid_starts_ptrs, heights, widths

    def grid_embedding(self, input_ids, input_token_positions, output_token_positions):
        """Generates grid positional embeddings."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        grid_embeds = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device)
        # for debug
        # input_example = input_ids[0][:15].tolist()
        # print(f"input_ids:{input_example}")
        # print('////////////////////////////////////')
        # input_token_positions = input_token_positions[:][:2]
        # output_token_positions = output_token_positions[:][:2]
        for b in range(batch_size):
            sample_ids = input_ids[b]
            # print(f"sample_ids:{sample_ids}")
            grid_starts_ptrs, heights, widths = self._get_grid_info(sample_ids, input_token_positions[b])
            g1, h1, w1 = self._get_grid_info(sample_ids, output_token_positions[b])
            grid_starts_ptrs.extend(g1)
            heights.extend(h1)
            widths.extend(w1)
            for p, idx in enumerate(grid_starts_ptrs):
            
                if heights[p] > 0 and widths[p] > 0:
                    grid_pos_embed = self._grid_embed(widths[p], heights[p], device)
                    # print(grid_pos_embed.shape)
                    # print(grid_embeds.shape)
                    for i in range(heights[p]):
                        start_pos = idx + i * (widths[p] + 1)
                        end_pos = start_pos + widths[p]
                        # print(f"endpose:{end_pos}")
                        # print(f"seq_len:{seq_len}")
                        if end_pos < seq_len:  # Ensure within bounds
                            grid_embeds[b, start_pos:end_pos] = grid_pos_embed[i, :,:]

        if self.if_dropout:
            grid_embeds = self.dropout(grid_embeds)
        return grid_embeds

    def forward(self, input_ids, input_token_positions, output_token_positions):
        """Computes the final embeddings by combining token and grid embeddings."""
        # input_ids = input_ids[:][0:15]
        # input_example = input_ids[0][:15].tolist()
        # print(f"input_ids:{input_example}")
        token_embeds = super().forward(input_ids)
        grid_embeds = self.grid_embedding(input_ids, input_token_positions, output_token_positions)
        return token_embeds + grid_embeds
#

    

  
class CustomLlamaModel(LlamaForCausalLM):
    def __init__(self, config, extra_size=None, **kwargs):
        super().__init__(config, **kwargs)
        custom_embedding = GridEmbedding(config, if_dropout=False, tokenizer=None, scaled_position_embedding=True)
        self.set_input_embeddings(custom_embedding)


    def forward(self, input_ids, labels, input_token_positions, output_token_positions,attention_mask=None, output_hidden_states=True, **kwargs):
        
        
        kwargs.pop("inputs_embeds", None)
        inputs_embeds = self.model.embed_tokens(input_ids, input_token_positions, output_token_positions) 
        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, output_hidden_states=True, **kwargs)
        
        # print(len(input_ids[0]))
        # if input_embeds == None:
        #     kwargs.pop("inputs_embeds", None)
        #     inputs_embeds = self.model.embed_tokens(input_ids, input_token_positions, output_token_positions) 
        #     return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, output_hidden_states=True, **kwargs)
        # else:
        #     return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, output_hidden_states=True, **kwargs)










