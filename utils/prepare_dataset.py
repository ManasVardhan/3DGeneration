import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class ShoeMultiViewDataset(Dataset):
    def __init__(self, data_root, transform=None):
        """
        Expected structure:
        data_root/
            shoe_001/
                front.png
                back.png
                left.png
                right.png
                top.png
                bottom.png
                mesh.obj (ground truth)
            shoe_002/
                ...
        """
        self.data_root = data_root
        self.shoe_ids = [d for d in os.listdir(data_root) 
                         if os.path.isdir(os.path.join(data_root, d))]
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.views = ['front', 'back', 'left', 'right', 'top', 'bottom']

    def __len__(self):
        return len(self.shoe_ids)

    def __getitem__(self, idx):
        shoe_id = self.shoe_ids[idx]
        shoe_path = os.path.join(self.data_root, shoe_id)

        # Load all views
        images = {}

        for view in self.views:
            img_path = os.path.join(shoe_path, f"{view}.png")
            img = Image.open(img_path).convert('RGB')
            images[view] = self.transform(img)

        # Load ground truth mesh path (for evaluation)
        mesh_path = os.path.join(shoe_path, "mesh.obj")

        return {
            'images': images,
            'mesh_path': mesh_path,
            'shoe_id': shoe_id
        }


# Create dataloader
dataset = ShoeMultiViewDataset('path/to/shoe_dataset')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)