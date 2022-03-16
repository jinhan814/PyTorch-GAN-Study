from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
class IdolDataset(Dataset):
    def __init__(self, dir, transform):
        self.dir_path = dir
        self.img_paths = os.listdir(self.dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # load image
        img_path = self.img_paths[idx]
        img = Image.open(os.path.join(self.dir_path,img_path)).convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
            

        img = self.transform(img)
        return img
