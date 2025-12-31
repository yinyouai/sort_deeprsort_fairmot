import torch
import torch.nn as nn
import torch.nn.functional as F

class DLA34Backbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super(DLA34Backbone, self).__init__()
        # 简化示例 backbone
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding

class FairMOTModel:
    def __init__(self, weight_path=None, embedding_dim=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DLA34Backbone(embedding_dim)
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def extract_embedding(self, frame_image, num_dets):
        """
        frame_image: torch tensor, shape (B,C,H,W) 或 None（占位）
        num_dets: 检测框数量
        返回每个检测框的 embedding
        """
        # 如果没有真实图像，返回随机 embedding
        if frame_image is None:
            embeddings = [torch.randn(128).numpy() for _ in range(num_dets)]
        else:
            with torch.no_grad():
                out = self.model(frame_image.to(self.device))
                embeddings = [o.cpu().numpy() for o in out]
        return embeddings
