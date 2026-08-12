import torch
import torch.nn as nn

from Forward_Warp import forward_warp


def get_disparity(depth, max_disp, width, max_disp_reference_width):
    effective_max_disp = max_disp * width / max_disp_reference_width
    return (depth * 2.0 - 1.0) * effective_max_disp


class ForwardWarpStereo(nn.Module):
    def __init__(self, eps=1e-6, return_occlusion_mask=True):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.return_occlusion_mask = return_occlusion_mask
        self.forward_warp = forward_warp()

    def forward(self, image, disparity):
        image = image.contiguous()
        disparity = disparity.contiguous()
        weights = 1.414 ** (disparity - disparity.min())
        horizontal_flow = -disparity.squeeze(1)
        vertical_flow = torch.zeros_like(horizontal_flow, requires_grad=False)
        flow = torch.stack((horizontal_flow, vertical_flow), dim=-1)
        result_accumulated = self.forward_warp(image * weights, flow)
        weights_accumulated = self.forward_warp(weights, flow)
        result = result_accumulated / weights_accumulated.clamp(min=self.eps)
        if not self.return_occlusion_mask:
            return result
        occlusion_mask = self.forward_warp(torch.ones_like(disparity), flow)
        occlusion_mask = 1.0 - occlusion_mask.clamp(0.0, 1.0)
        return result, occlusion_mask
