from torch.utils.data import Dataset
import os
from PIL import Image
class IdolDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir_path = dir
        self.img_paths = os.listdir(self.dir)
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # load image
        img_path = self.img_paths[idx]
        img = Image.open(os.path.join(self.dir_path,img_path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img
