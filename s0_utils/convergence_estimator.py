import torch
import torch.nn.functional as F
import os
import logging

import s0_utils.global_params as g
from s0_utils.u2net import U2NETP

logger = logging.getLogger(__name__)


class ConvergenceEstimator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConvergenceEstimator, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path=None, device=None):
        if hasattr(self, "model"):
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_path is None:
            model_path = str(g.CONVERGENCE_WEIGHTS_PATH)
        if not os.path.exists(model_path):
            logger.warning(f"Convergence model not found at {model_path}")
            self.model = None
            return
        try:
            self.model = U2NETP(in_ch=6, out_ch=1)
            logger.info(f"Loading convergence model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("u2netp."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
            self.model.to(self.device).eval()
            logger.info("Convergence model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load convergence model: {e}")
            self.model = None

    def preprocess(self, rgb_tensor, depth_tensor):
        rgb_small = F.interpolate(
            rgb_tensor, (192, 192), mode="bilinear", align_corners=False
        )
        depth_small = F.interpolate(
            depth_tensor, (192, 192), mode="bilinear", align_corners=False
        )
        depth_small = torch.clamp(depth_small, 0.0, 1.0)
        depth_sqrt = torch.pow(depth_small, 0.5)
        depth_pow = torch.pow(depth_small, 2.0)
        x = torch.cat([rgb_small, depth_small, depth_sqrt, depth_pow], dim=1)
        return x, depth_small

    def calculate_robust_depth(
        self, saliency_map, depth_small, user_convergence_ratio=0.5
    ):
        batch_size = depth_small.shape[0]
        results = []
        for i in range(batch_size):
            mask = saliency_map[i, 0] > 0.5
            d_vals = depth_small[i, 0][mask]
            if d_vals.numel() == 0:
                results.append(0.5)
                continue
            q01 = torch.quantile(d_vals, 0.1)
            q09 = torch.quantile(d_vals, 0.9)
            center = (q01 + q09) / 2.0
            obj_range = q09 - q01
            if obj_range < 1e-6:
                q_pos = q01
            else:
                expanded_range = obj_range * 3.0
                q_pos = center + (user_convergence_ratio - 0.5) * expanded_range
            q_pos = torch.clamp(q_pos, 0.0, 1.0)
            results.append(q_pos.item())
        return results

    @torch.inference_mode()
    def predict(self, rgb_tensor, depth_tensor, user_ratio=0.5):
        if self.model is None:
            return [0.5] * rgb_tensor.shape[0]
        rgb_tensor = rgb_tensor.to(self.device)
        depth_tensor = depth_tensor.to(self.device)
        input_tensor, depth_small = self.preprocess(rgb_tensor, depth_tensor)
        outputs = self.model(input_tensor)
        saliency = outputs[0]
        return self.calculate_robust_depth(saliency, depth_small, user_ratio)
