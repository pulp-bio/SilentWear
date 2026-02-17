
import sys
from pathlib import Path
import yaml
from itertools import product
import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from einops import rearrange, repeat
import torch.nn.functional as F
PROJECT_ROOT = Path(__file__).resolve().parents[2]   
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
from models.utils import count_params
from utils.general_utils import load_yaml
from types import SimpleNamespace

from torchinfo import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_namespace(d: dict) -> SimpleNamespace:
    """
    Turns nested dicts into attribute-access objects:
    cfg.num_filters_per_block instead of cfg["num_filters_per_block"].
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [to_namespace(x) for x in d]
    return d


def validate_tcnformer_kwargs(kwargs: dict) -> None:
    # Basic shape consistency checks to catch config mistakes early
    num_levels = len(kwargs["num_filters_per_block"])
    ncpb = int(kwargs["num_convs_per_block"])

    for key in ["kernel_sizes", "strides", "dilations"]:
        if len(kwargs[key]) != num_levels:
            raise ValueError(f"{key} must have length {num_levels} (one per block). Got {len(kwargs[key])}.")

        for i in range(num_levels):
            if len(kwargs[key][i]) != ncpb:
                raise ValueError(
                    f"{key}[{i}] must have length {ncpb} (one per conv in block). "
                    f"Got {len(kwargs[key][i])}."
                )
            
class Chomp1d(nn.Module):
    """Causal padding removal for temporal convolution"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalConvLayer(nn.Module):
    """Single temporal convolution block with chomp, activation and dropout"""
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, dilation, dropout, is_batchnorm):
        super().__init__()
        
        # Causal convolution
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=(kernel_size-1)*dilation,
            dilation=dilation
        ))
        self.chomp = Chomp1d((kernel_size-1)*dilation) if stride == 1 else None
        self.batchnorm = nn.BatchNorm1d(out_channels) if is_batchnorm else None
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.conv(x)
        if self.chomp is not None:
            x = self.chomp(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        x = self.activation(x)
        return self.dropout(x)

class TemporalBlock(nn.Module):
    """Temporal block containing multiple convolutional layers with residual connection"""
    def __init__(self, num_convs, in_channels, out_channels, 
                 kernel_sizes, strides, dilations, dropout, is_batchnorm):
        super().__init__()
        
        # Create convolutional layers
        self.convs = nn.ModuleList()
        current_in = in_channels
        for i in range(num_convs):
            self.convs.append(TemporalConvLayer(
                current_in, out_channels, kernel_sizes[i],
                strides[i], dilations[i], dropout, is_batchnorm
            ))
            current_in = out_channels  # Subsequent layers use out_channels as input

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        for conv in self.convs:
            x = conv(x)
        return self.activation(x + residual)


class WindowedTransformer(nn.Module):
    """Transformer operating on fixed-length temporal windows"""
    def __init__(self, input_dim, window_size, num_heads, num_layers, 
                 dim_feedforward=2048, dropout=0.1, positional_encoding=True):
        super().__init__()
        self.window_size = window_size
        self.positional_encoding = positional_encoding
        
        # Positional encoding
        if positional_encoding:
            self.pos_encoder = nn.Embedding(window_size, input_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch*nw, window_size, input_dim)
        if self.positional_encoding:
            positions = torch.arange(self.window_size, device=x.device).expand(x.size(0), self.window_size)
            x = x + self.pos_encoder(positions)
        
        # Apply transformer with mask to prevent cross-window attention
        return self.transformer(x)

class TCNformer(nn.Module):
    """Temporal Convolutional Network with Transformer window processing"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Build temporal blocks
        self.temporal_blocks = self._build_temporal_blocks()
        
        # Build Transformer and classification layers
        self.transformer = WindowedTransformer(
            input_dim=config.num_filters_per_block[-1],
            window_size=config.transformer_window_size,
            num_heads=config.transformer_num_heads,
            num_layers=config.transformer_num_layers,
            dim_feedforward=config.transformer_dim_feedforward,
            dropout=config.transformer_dropout,
            positional_encoding=config.use_positional_encoding
        )
        
        self.classifier = nn.Linear(config.num_filters_per_block[-1], config.num_classes)
        
        # Calculate initial receptive field
        self.receptive_field = self._calculate_receptive_field()

    def _build_temporal_blocks(self):
        blocks = nn.ModuleList()
        num_levels = len(self.config.num_filters_per_block)
        
        for i in range(num_levels):
            in_channels = self.config.input_size if i == 0 \
                else self.config.num_filters_per_block[i-1]
            
            blocks.append(TemporalBlock(
                num_convs=self.config.num_convs_per_block,
                in_channels=in_channels,
                out_channels=self.config.num_filters_per_block[i],
                kernel_sizes=self.config.kernel_sizes[i],
                strides=self.config.strides[i],
                dilations=self.config.dilations[i],
                dropout=self.config.dropout,
                is_batchnorm=self.config.is_batchnorm
            ))
        
        return nn.Sequential(*blocks)
    
    def _calculate_receptive_field(self):
        """Calculate theoretical receptive field size including transformer window"""
        rf = 1
        accumulated_stride = 1
        
        # TCN contribution
        for block_idx in range(len(self.config.dilations)):
            for conv_idx in range(self.config.num_convs_per_block):
                kernel = self.config.kernel_sizes[block_idx][conv_idx]
                dilation = self.config.dilations[block_idx][conv_idx]
                stride = self.config.strides[block_idx][conv_idx]
                
                rf += (kernel - 1) * dilation * accumulated_stride
                accumulated_stride *= stride
        
        # Add transformer window contribution
        rf += self.config.transformer_window_size - 1
        
        return rf

    def forward(self, x, return_latent=False, is_latent=False, latent_stage="input"):
        """
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor. If `is_latent=True`, this is a latent representation from `latent_stage`.
        return_latent : bool
            Whether to return latent representation at `latent_stage`.
        is_latent : bool
            If True, input `x` is already a latent (skip processing before `latent_stage`).
        latent_stage : str
            Defines where `x` is extracted from (when `is_latent=True`) or where to extract (when `return_latent=True`).
            Options:
            - "input" → raw input
            - "post_temporal" → after temporal blocks
            - "post_transformer" → after transformer (default)
        """

        # --------------------------
        # Case 1: Input is raw data (is_latent=False)
        # --------------------------

        if not is_latent:

            # Return raw input as latent (for input replay)
            if return_latent and latent_stage == "input":
                return x
            
            # Temporal blocks processing
            x = self.temporal_blocks(x)
            
            # Return latent after temporal blocks
            if return_latent and latent_stage == "post_temporal":
                return x
            
            # Remove initial receptive field portion (post-temporal processing)
            x = x[:, :, self.receptive_field:]

            # Prepare for transformer processing
            batch_size, channels, seq_len = x.size()
            
            # Prepare for transformer: (batch, channels, seq_len) → (batch, seq_len, channels)
            x = x.permute(0, 2, 1)
            
            # Padding for windowing
            window_size = self.config.transformer_window_size
            pad = (window_size - (seq_len % window_size)) % window_size
            x = nn.functional.pad(x, (0, 0, 0, pad))  # pad sequence length dim
            padded_seq_len = seq_len + pad
            
            # Split into windows: (batch, padded_seq_len, channels) → (batch*num_windows, window_size, channels)
            x = rearrange(x, 'b (nw w) c -> (b nw) w c', w=window_size)
            
            # Transformer processing
            x = self.transformer(x)
            
            # Reconstruct sequence: (batch*num_windows, window_size, channels) → (batch, padded_seq_len, channels)
            x = rearrange(x, '(b nw) w c -> b (nw w) c', b=batch_size)
            x = x[:, :seq_len, :]  # Remove padding

            # Return latent after transformer
            if return_latent and latent_stage == "post_transformer":
                return x
            

            
            # Final output shape: (batch, num_classes, seq_len)

            ######### MODIFIED to return one output
            # Classifier processing
            #x = self.classifier(x)
            # x.permute(0, 2, 1)

            #########################
            x = self.classifier(x.mean(dim=1))  # (B, C) -> (B, num_classes)
            return x
        
        # --------------------------
        # Case 2: Input is latent (is_latent=True)
        # --------------------------
        else:
            # Check valid latent_stage
            valid_stages = ["input", "post_temporal", "post_transformer"]
            if latent_stage not in valid_stages:
                raise ValueError(f"Invalid latent_stage {latent_stage}. Must be one of {valid_stages}")
            
            # --------------------------
            # Latent is "input" (raw input) → process from scratch
            # --------------------------
            if latent_stage == "input":
                # Equivalent to is_latent=False, start from raw input
                return self.forward(x, return_latent=False, is_latent=False, latent_stage=latent_stage)
            
            # --------------------------
            # Latent is "post_temporal" → continue after temporal blocks
            # --------------------------
            elif latent_stage == "post_temporal":
                # x shape: (batch, channels, seq_len) (output of temporal_blocks)
                
                # Remove initial receptive field portion
                x = x[:, :, self.receptive_field:]

                
                # Prepare for transformer processing
                batch_size, channels, seq_len = x.size()

                x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
                
                # Padding for windowing
                window_size = self.config.transformer_window_size
                pad = (window_size - (seq_len % window_size)) % window_size
                x = nn.functional.pad(x, (0, 0, 0, pad))
                padded_seq_len = seq_len + pad
                
                # Split into windows
                x = rearrange(x, 'b (nw w) c -> (b nw) w c', w=window_size)
                
                # Transformer processing
                x = self.transformer(x)
                
                # Reconstruct sequence
                x = rearrange(x, '(b nw) w c -> b (nw w) c', b=batch_size)
                x = x[:, :seq_len, :]  # Remove padding
                
                # Classifier processing
                x = self.classifier(x)
                
                # Final output
                return x.permute(0, 2, 1)
            
            # --------------------------
            # Latent is "post_transformer" → continue after transformer
            # --------------------------
            elif latent_stage == "post_transformer":
                # Classifier processing
                x = self.classifier(x)
                
                # Final output
                return x.permute(0, 2, 1)




if __name__ == "__main__":
    # -----------------------------
    # Settings
    # -----------------------------
    cfg_path = PROJECT_ROOT / "config" / "models_configs/tcnformer_base.yaml"  
    if cfg_path.exists() == False:
        print(f"Could not find config file: {cfg_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy input sizes (you can also read these from cfg if you store them there)
    batch_size = 32
    num_emg_channels = 14
    num_time_samples = 700
    num_classes = 8

    # -----------------------------
    # Load YAML
    # -----------------------------
    raw_cfg = load_yaml(cfg_path)
    
    print(raw_cfg)

    model_cfg = raw_cfg["model"]
    kwargs = model_cfg["kwargs"]
    # Inject required fields for the model
    kwargs["input_size"] = num_emg_channels
    kwargs["num_classes"] = num_classes

    # Optional: validate structure for your TCNformer
    validate_tcnformer_kwargs(kwargs)

    # Build the attribute-style config your TCNformer expects
    config_obj = to_namespace(kwargs)


    # -----------------------------
    # Build model
    # -----------------------------
    model = TCNformer(config_obj).to(device)
    model.eval()

    # -----------------------------
    # Params
    # -----------------------------
    total, trainable = count_params(model)
    print(f"Model: {model_cfg.get('name', 'tcnformer')}")
    print(f"Total params: {total:,}")
    print(f"Trainable:    {trainable:,}")

    # -----------------------------
    # Torchinfo summary
    # -----------------------------
    dummy = torch.rand(batch_size, num_emg_channels, num_time_samples, device=device)

    # For sequence models, torchinfo works best with input_data
    print("\nTorchinfo summary:\n")
    summary(
        model,
        input_data=dummy,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        depth=4,
        verbose=1,
    )

    # -----------------------------
    # Quick forward sanity check
    # -----------------------------
    with torch.no_grad():
        y = model(dummy)
    print("\nOutput shape:", tuple(y.shape))
