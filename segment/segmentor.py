import torch


class Segmentor:
    def __init__(self, gpu_org_img, gpu_mask_img):

        self.gpu_org_img = gpu_org_img
        self.gpu_mask_img = gpu_mask_img

    def remove_background(self):
        gpu_segment_mask = self.gpu_mask_img.gt(0.)
        gpu_segment_mask3 = torch.cat((gpu_segment_mask, gpu_segment_mask, gpu_segment_mask))
        gpu_mask3 = torch.cat((self.gpu_mask_img, self.gpu_mask_img, self.gpu_mask_img))
        gpu_mask3[gpu_segment_mask3] = self.gpu_org_img[gpu_segment_mask3]
        gpu_mask3 = gpu_mask3.flip(dims=[0])
        gpu_mask3 = gpu_mask3.permute(1, 2, 0).contiguous()
        cpu_mask3 = gpu_mask3.cpu().numpy()
        return cpu_mask3
