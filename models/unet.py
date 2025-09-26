import torch
import torch.nn as nn


class ConvDropoutNormNonlin(nn.Module):
    """
    Convolutional layer with dropout, normalization, and non-linearity.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)

        if self.dropout_op is not None and self.dropout_op_kwargs["p"] is not None and self.dropout_op_kwargs["p"] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.norm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.activation = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.activation(self.norm(x))


class MultiLayerConvDropoutNormNonlin(nn.Module):
    """
    Multi-layer convolutional block with configurable number of layers.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        num_layers=2,
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        assert num_layers >= 1, "Number of layers must be at least 1, got {}".format(num_layers)
        self.num_layers = num_layers

        self.conv1 = ConvDropoutNormNonlin(
            input_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        for layer in range(2, num_layers + 1):
            setattr(
                self,
                f"conv{layer}",
                ConvDropoutNormNonlin(
                    output_channels,
                    output_channels,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                ),
            )

    def forward(self, x):
        x = self.conv1(x)
        for layer in range(2, self.num_layers + 1):
            x = getattr(self, f"conv{layer}")(x)

        return x

    @staticmethod
    def get_block_constructor(n_layers):
        def _block(input_channels, output_channels, **kwargs):
            return MultiLayerConvDropoutNormNonlin(input_channels, output_channels, num_layers=n_layers, **kwargs)

        return _block


class UNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        starting_filters: int = 64,
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        weightInitializer=None,
        basic_block=MultiLayerConvDropoutNormNonlin,
        transformer_encoder=False,
        transformer_encoder_kwargs={
            "d_model": 512,
            "nhead": 8,
            "dim_feedforward": 2048,
            "dropout": 0.1,
        },
    ) -> None:
        super().__init__()

        # Task specific
        self.filters = starting_filters

        # Model parameters
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.weightInitializer = weightInitializer
        self.basic_block = basic_block
        self.transformer_encoder = transformer_encoder
        self.pool_op = nn.MaxPool3d

        self.in_conv = self.basic_block(
            input_channels=input_channels,
            output_channels=self.filters,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool1 = self.pool_op(2)
        self.encoder_conv1 = self.basic_block(
            input_channels=self.filters,
            output_channels=self.filters * 2,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool2 = self.pool_op(2)
        self.encoder_conv2 = self.basic_block(
            input_channels=self.filters * 2,
            output_channels=self.filters * 4,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool3 = self.pool_op(2)
        self.encoder_conv3 = self.basic_block(
            input_channels=self.filters * 4,
            output_channels=self.filters * 8,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool4 = self.pool_op(2)
        self.encoder_conv4 = self.basic_block(
            input_channels=self.filters * 8,
            output_channels=self.filters * 16,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, x):
        x0 = self.in_conv(x)

        x1 = self.pool1(x0)
        x1 = self.encoder_conv1(x1)

        x2 = self.pool2(x1)
        x2 = self.encoder_conv2(x2)

        x3 = self.pool3(x2)
        x3 = self.encoder_conv3(x3)

        x4 = self.pool4(x3)
        x4 = self.encoder_conv4(x4)

        return [x0, x1, x2, x3, x4]


def unet_encoder(
    input_channels: int = 1,
    starting_filters: int = 64,
):
    """
    UNet Encoder with configurable filter size.
    
    Args:
        input_channels: Number of input channels
        starting_filters: Number of starting filters (32 for B, 64 for XL)
        
    Returns:
        UNet encoder with specified starting_filters
    """
    encoder = UNetEncoder(
        input_channels=input_channels,
        starting_filters=starting_filters,
    )
    return encoder


def unet_b_encoder(
    input_channels: int = 1,
):
    return unet_encoder(input_channels=input_channels, starting_filters=32)


def unet_xl_encoder(
    input_channels: int = 1,
):
    return unet_encoder(input_channels=input_channels, starting_filters=64)

    
if __name__ == "__main__":
    # Test both configurations
    print("Testing UNet Encoder configurations...")
    
    # Create encoders
    encoder_b = unet_b_encoder(input_channels=1)
    encoder_xl = unet_xl_encoder(input_channels=1)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 1, 64, 64, 64)  # Batch, Channels, Depth, Height, Width
    
    # Test B encoder
    encoded_features_b = encoder_b(dummy_input)
    print(f"\nUNet B Encoder (32 filters):")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Number of encoded feature maps: {len(encoded_features_b)}")
    for i, feat in enumerate(encoded_features_b):
        print(f"Feature map {i} shape: {feat.shape}")
    print(f"Encoder parameters: {sum(p.numel() for p in encoder_b.parameters()):,}")
    
    # Test XL encoder
    encoded_features_xl = encoder_xl(dummy_input)
    print(f"\nUNet XL Encoder (64 filters):")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Number of encoded feature maps: {len(encoded_features_xl)}")
    for i, feat in enumerate(encoded_features_xl):
        print(f"Feature map {i} shape: {feat.shape}")
    print(f"Encoder parameters: {sum(p.numel() for p in encoder_xl.parameters()):,}")
    

    # Comparison
    print(f"\nComparison:")
    print(f"UNet B Encoder: {sum(p.numel() for p in encoder_b.parameters()):,} parameters")
    print(f"UNet XL Encoder: {sum(p.numel() for p in encoder_xl.parameters()):,} parameters")
