import argparse
import sys
import os
import math
import warnings
from typing import Callable, Optional, Tuple, Union, List, Any, Dict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# ============================================================================
# EMBEDDED CONFIGURATIONS
# ============================================================================

DEFAULT_CONFIG = {
    "MODEL": {"WEIGHTS": ""},
    "compute_precision": {
        "grad_scaler": True,
        "teacher": {
            "backbone": {
                "sharding_strategy": "SHARD_GRAD_OP",
                "mixed_precision": {
                    "param_dtype": "fp16",
                    "reduce_dtype": "fp16",
                    "buffer_dtype": "fp32",
                },
            }
        },
    },
    "student": {
        "arch": "vit_large_3d",
        "patch_size": 16,
        "drop_path_rate": 0.3,
        "layerscale": 1.0e-05,
        "drop_path_uniform": True,
        "pretrained_weights": "",
        "full_pretrained_weights": "",
        "ffn_layer": "mlp",
        "block_chunks": 4,
        "qkv_bias": True,
        "proj_bias": True,
        "ffn_bias": True,
    },
    "teacher": {
        "momentum_teacher": 0.992,
        "final_momentum_teacher": 1,
        "warmup_teacher_temp": 0.04,
        "teacher_temp": 0.07,
        "warmup_teacher_temp_epochs": 30,
    },
    "crops": {"global_crops_size": 112, "local_crops_size": 64},
}

# Model architecture configurations
MODEL_CONFIGS = {
    "vit_large_3d": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "patch_size": 16,
        "in_chans": 1,
    }
}


# ============================================================================
# EMBEDDED MODEL COMPONENTS
# ============================================================================


def make_3tuple(x):
    """Convert input to 3-tuple."""
    if isinstance(x, tuple):
        assert len(x) == 3
        return x
    assert isinstance(x, int)
    return (x, x, x)


class Identity(nn.Module):
    """Identity layer."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class LayerScale(nn.Module):
    """Layer scale module."""

    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    """MLP module."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Standard attention module."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(nn.Module):
    """Memory efficient attention (fallback to standard attention)."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.attention = Attention(
            dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop
        )

    def forward(self, x):
        return self.attention(x)


class Block(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias=True,
        ffn_bias=True,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ffn_layer="mlp",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MemEffAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else Identity()

        self.norm2 = norm_layer(dim)
        if ffn_layer == "mlp":
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=drop,
                bias=ffn_bias,
            )
        else:
            raise ValueError(f"Unsupported ffn_layer: {ffn_layer}")
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed3d(nn.Module):
    """3D image to patch embedding."""

    def __init__(
        self,
        img_size=96,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        norm_layer=None,
        flatten_embedding=True,
    ):
        super().__init__()
        image_HW = make_3tuple(img_size)
        patch_HW = make_3tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
            image_HW[2] // patch_HW[2],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1] * patch_grid_size[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def forward(self, x):
        _, _, H, W, D = x.shape
        patch_H, patch_W, patch_D = self.patch_size

        assert H % patch_H == 0, (
            f"Input image height {H} is not a multiple of patch height {patch_H}"
        )
        assert W % patch_W == 0, (
            f"Input image width {W} is not a multiple of patch width: {patch_W}"
        )
        assert D % patch_D == 0, (
            f"Input image depth {D} is not a multiple of patch depth: {patch_D}"
        )

        x = self.proj(x)  # B C H W D
        H, W, D = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)  # B HWD C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, D, self.embed_dim)  # B H W D C
        return x


class BlockChunk(nn.ModuleList):
    """Block chunk for FSDP wrapping."""

    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer3d(nn.Module):
    """3D DINO Vision Transformer."""

    def __init__(
        self,
        img_size=112,
        patch_size=16,
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed3d,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=0.0,
                attn_drop=0.0,
                init_values=init_values,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                ffn_layer=ffn_layer,
            )
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)
        self.head = Identity()

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # Return only cls token


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def normalize_image(image_tensor, percentile_range=(0.0005, 0.9995)):
    """Normalize image tensor to range [-1, 1] using percentile-based normalization."""
    min_percentile, max_percentile = percentile_range
    min_val = torch.quantile(image_tensor, min_percentile)
    max_val = torch.quantile(image_tensor, max_percentile)

    # Normalize to [0, 1] then scale to [-1, 1]
    normalized = (image_tensor - min_val) / (max_val - min_val)
    normalized = torch.clip(normalized * 2 - 1, -1, 1)

    return normalized


def build_model_for_eval(config, pretrained_weights=None):
    """Build model for evaluation."""
    model_config = MODEL_CONFIGS["vit_large_3d"]

    model = DinoVisionTransformer3d(
        img_size=config.get("crops", {}).get("global_crops_size", 112),
        patch_size=model_config["patch_size"],
        in_chans=model_config["in_chans"],
        embed_dim=model_config["embed_dim"],
        depth=model_config["depth"],
        num_heads=model_config["num_heads"],
        mlp_ratio=model_config["mlp_ratio"],
        qkv_bias=config.get("student", {}).get("qkv_bias", True),
        ffn_bias=config.get("student", {}).get("ffn_bias", True),
        proj_bias=config.get("student", {}).get("proj_bias", True),
        drop_path_rate=config.get("student", {}).get("drop_path_rate", 0.0),
        drop_path_uniform=config.get("student", {}).get("drop_path_uniform", False),
        init_values=config.get("student", {}).get("layerscale", None),
        ffn_layer=config.get("student", {}).get("ffn_layer", "mlp"),
        block_chunks=config.get("student", {}).get("block_chunks", 1),
    )

    if pretrained_weights and os.path.exists(pretrained_weights):
        try:
            checkpoint = torch.load(pretrained_weights, map_location="cpu")
            if "teacher" in checkpoint:
                state_dict = checkpoint["teacher"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from: {pretrained_weights}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Using random initialization!")
    else:
        print("No weights found, using random initialization!")

    model.eval()
    return model


def run_inference(model, input_tensor, device="cuda"):
    """Run inference on input tensor."""
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
    else:
        model = model.cpu()
        input_tensor = input_tensor.cpu()
        device = "cpu"

    with torch.no_grad():
        output = model(input_tensor)

    return output


# ============================================================================
# MAIN SCRIPT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Standalone 3DINO-ViT Inference Script"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pretrained weights (optional)",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="1,1,112,112,112",
        help="Input tensor shape as comma-separated values (default: 1,1,112,112,112)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save features (optional)",
    )
    parser.add_argument(
        "--random_input",
        action="store_true",
        help="Use random input tensor instead of loading from file",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input .npy or .pt file (if not using random input)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=112,
        help="Image size for model (default: 112)",
    )

    args = parser.parse_args()

    # Parse input shape
    try:
        input_shape = tuple(map(int, args.input_shape.split(",")))
        if len(input_shape) != 5:
            raise ValueError(
                "Input shape must have 5 dimensions: (batch, channels, depth, height, width)"
            )
    except ValueError as e:
        print(f"Error parsing input shape: {e}")
        sys.exit(1)

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Input shape: {input_shape}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.img_size}")

    # Update config with image size
    config = DEFAULT_CONFIG.copy()
    config["crops"]["global_crops_size"] = args.img_size

    # Load model
    try:
        print("Building model...")
        model = build_model_for_eval(config, args.weights)
        print(
            f"Model built successfully with {sum(p.numel() for p in model.parameters())} parameters"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Prepare input
    if args.random_input:
        print("Using random input tensor")
        input_tensor = torch.randn(input_shape)
    elif args.input_file:
        print(f"Loading input from: {args.input_file}")
        try:
            if args.input_file.endswith(".npy"):
                input_tensor = torch.from_numpy(np.load(args.input_file))
            elif args.input_file.endswith(".pt"):
                input_tensor = torch.load(args.input_file)
            else:
                print("Unsupported file format. Use .npy or .pt files")
                sys.exit(1)

            # Ensure correct shape
            if input_tensor.shape != input_shape:
                print(
                    f"Warning: Input shape {input_tensor.shape} doesn't match expected {input_shape}"
                )
                print("Reshaping input tensor...")
                input_tensor = input_tensor.view(input_shape)
        except Exception as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)
    else:
        print(
            "Using random input tensor (use --random_input or --input_file to specify)"
        )
        input_tensor = torch.randn(input_shape)

    # Normalize input
    print("Normalizing input...")
    input_tensor = normalize_image(input_tensor)
    print(f"Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

    # Run inference
    print("Running inference...")
    try:
        output = run_inference(model, input_tensor, args.device)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

    # Save output if requested
    if args.output:
        try:
            if args.output.endswith(".npy"):
                np.save(args.output, output.cpu().numpy())
            elif args.output.endswith(".pt"):
                torch.save(output.cpu(), args.output)
            else:
                # Default to .npy
                np.save(args.output + ".npy", output.cpu().numpy())
                args.output += ".npy"
            print(f"Features saved to: {args.output}")
        except Exception as e:
            print(f"Error saving output: {e}")

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
