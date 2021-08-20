import torch,cv2,os
from torch.utils.data import Dataset, DataLoader
import numpy as np

TRAIN = 'dataset/train'
MASK = 'dataset/label'


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class MyDataset(Dataset):
    def __init__(self, tfms=None, n_class=5):
        self.fnames = [fname for fname in os.listdir(TRAIN)]
        self.tfms = tfms
        self.n_class = n_class
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN, fname)), cv2.COLOR_BGR2RGB)
        mask_origin = cv2.cvtColor(cv2.imread(os.path.join(MASK, fname)), cv2.COLOR_BGR2GRAY)
        #print(mask_origin.min(),mask_origin.max())
        mask = np.zeros((mask_origin.shape[0],mask_origin.shape[1],self.n_class), dtype=np.float)
        for i in range(self.n_class):
            mask[:,:,i] = mask_origin==i
        # print(mask.max())
        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
            img = img / 255.0
        return img2tensor((img - img.mean()) / img.std()), img2tensor(mask)

if __name__=='__main__':
    data = MyDataset()
    img,label = data.__getitem__(100)
    print(img.shape,label.shape)
