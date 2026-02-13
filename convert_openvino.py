'''
    Model should be trained without normalization
'''


import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import openvino as ov

from utils import load_model_from_state_dict

logger = logging.getLogger(__name__)


class VQVAEWrapper(nn.Module):
    """Wrapper that loads a trained AE and exposes a simple forward.

    Args:
        model_path: Path to a saved ``state_dict`` (.pth).
        in_channels: Input channels expected by the AE.
        out_channels: Output channels produced by the AE.
        device: Device to load and run the model on (e.g., "cpu", "cuda").
    """
    def __init__(self, model_path: str, device: str = 'cpu') -> None:
        super().__init__()
        state_dict = torch.load(model_path, weights_only=False, map_location=device)
        self.model, _ = load_model_from_state_dict(
            state_dict, 'vqvae.VQVAE'
        )
        self.model.eval()
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass without gradients.

        Args:
            x: Input tensor of shape ``[N, C, H, W]``.

        Returns:
            Model output tensor.
        """
        with torch.no_grad():
            output = self.model(x.to(self.device))
        return output[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for OpenVINO conversion.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Convert a trained UNet denoiser to OpenVINO IR")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained PyTorch weights (.pth)")
    parser.add_argument("--output", type=str, required=True, help="Path to save OpenVINO IR XML (e.g., openvino_models/denoiser.xml)")
    parser.add_argument("--img-size", type=int, default=256, help="Input image size (H=W)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for example input")
    parser.add_argument("--device", type=str, default='cpu', help="Device for loading weights (cpu/cuda)")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Convert a trained AE to OpenVINO IR and save it.

    Args:
        args: Parsed command-line arguments.
    """
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger.info("Converting weights: %s", args.weights)

    # Wrap model and create example input
    vqvae = VQVAEWrapper(args.weights, device=args.device)
    example_input = torch.randn(args.batch, 3, args.img_size, args.img_size)

    # Convert the PyTorch model to OpenVINO IR
    ov_model = ov.convert_model(input_model=vqvae, example_input=example_input)

    # Add a preprocessing pipeline
    prep = ov.preprocess.PrePostProcessor(ov_model)
    prep.input().tensor() \
        .set_element_type(ov.Type.u8) \
        .set_layout(ov.Layout('NCHW')) \
        .set_color_format(ov.preprocess.ColorFormat.RGB) \
        .set_spatial_static_shape(args.img_size, args.img_size)

    # resize and simple scale
    ip = prep.input().preprocess()
    ip.convert_color(ov.preprocess.ColorFormat.RGB)
    ip.resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR, args.img_size, args.img_size) \
      .convert_element_type(ov.Type.f32) \
      .scale(255.)

    model_with_preprocess = prep.build()

    # Save the final model with embedded preprocessing
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model_with_preprocess, str(out_path))
    logger.info("VQVAE model saved to %s", str(out_path))


if __name__ == '__main__':
    main(parse_args())
