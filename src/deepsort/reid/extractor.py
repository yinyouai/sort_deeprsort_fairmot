import torch
import cv2
import numpy as np
from .model import Net  # 保持原项目的 Net


class Extractor:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net()
        self.model.to(self.device)
        self.model.eval()

        # 加载 checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # 取出 net_dict
        if 'net_dict' in checkpoint:
            state_dict = checkpoint['net_dict']
        else:
            state_dict = checkpoint

        # 处理 state_dict 前缀问题（有些 ckpt 权重没有 base. 前缀）
        new_state_dict = {}
        for k, v in state_dict.items():
            # 如果当前模型的 key 在 state_dict 中有对应的 key，直接保留
            if k in self.model.state_dict() and v.size() == self.model.state_dict()[k].size():
                new_state_dict[k] = v
            # 尝试去掉 'base.' 前缀匹配
            elif k.startswith('base.') and k[5:] in self.model.state_dict() and v.size() == self.model.state_dict()[
                k[5:]].size():
                new_state_dict[k[5:]] = v

        # 加载权重
        self.model.load_state_dict(new_state_dict, strict=False)

    def __call__(self, imgs):
        """
        imgs: list of numpy images (H, W, C), BGR format
        return: np.array of shape (N, feature_dim)
        """
        imgs_torch = []
        for img in imgs:
            # BGR -> RGB, resize, HWC -> CHW
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            imgs_torch.append(torch.from_numpy(img))

        imgs_torch = torch.stack(imgs_torch).to(self.device)
        with torch.no_grad():
            features = self.model(imgs_torch)
        return features.cpu().numpy()
