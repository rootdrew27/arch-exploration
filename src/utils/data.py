from torch.utils.data import Dataset
from pathlib import Path
import os

class RefNetDataset(Dataset):
    def __init__(self, label_dir:Path, image_dir:Path, transform=None, target_transform=None):
        
        img_files = [f for f in image_dir.iterdir()] 
        lbl_files = [f for f in label_dir.iterdir()]
        
        cur_ref = None
        inp_lbl_sets = []

        for img, lbl in zip(img_files, lbl_files):
            if os.path.getsize(lbl) == 0: # label file is empty (i.e. a background image)
                cur_ref = img 
            else:
                inp_lbl_sets.append((cur_ref.name, img.name, lbl.name))
        
        self.inp_lbl_sets = inp_lbl_sets 
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.inp_lbl_sets)
    
    def __getitem__(self, idx):
        ref, cur, lbl = self.inp_lbl_sets[idx]
        
        ref_img = os.path.join(self.image_dir, ref)
        cur_img = os.path.join(self.image_dir, cur)
        label = os.path.join(self.label_dir, lbl)
        
        if self.transform:
            cur_img = self.transform(cur_img)
            ref_img = self.transform(ref_img)
        if self.target_transform:
            label = self.target_transform(label)
            
        return ref_img, cur_img, label