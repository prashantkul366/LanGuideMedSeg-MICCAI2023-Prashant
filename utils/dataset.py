import json
import os
import torch
import pandas as pd
# from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,RandCoarseShuffled,RandRotated,RandZoomd,
#                               Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from monai.transforms import (Compose, NormalizeIntensityd,
                              RandZoomd, Resized,
                              ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class QaTa(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):

        super(QaTa, self).__init__()

        self.mode = mode

        # with open(csv_path, 'r') as f:
        #     self.data = pd.read_csv(f)
        if csv_path.endswith(".xlsx"):
            self.data = pd.read_excel(csv_path)
        else:
            self.data = pd.read_csv(csv_path)
        
        self.data.columns = self.data.columns.str.strip()
        print("Detected Columns:", self.data.columns.tolist())
        # self.image_list = list(self.data['Image'])
        # self.caption_list = list(self.data['Description'])

        # KVASIR UNCOMMENT THIS
        # self.image_list = list(self.data['image_name'])
        # self.caption_list = list(self.data['prompt_text'])

        # BUSI UNCOMMENT THIS
        # self.image_list = list(self.data['Filename'])
        # self.caption_list = list(self.data['Text'])

        cols = [c.lower() for c in self.data.columns]

        if 'filename' in cols:
            self.data.columns = self.data.columns.str.lower()
            self.image_list = list(self.data['filename'])
            self.caption_list = list(self.data['text'])

        elif 'image_name' in cols:
            self.data.columns = self.data.columns.str.lower()
            self.image_list = list(self.data['image_name'])
            self.caption_list = list(self.data['prompt_text'])

        else:
            raise ValueError("Unsupported dataset format")

        # if mode == 'train':
        #     self.image_list = self.image_list[:int(0.8*len(self.image_list))]
        #     self.caption_list = self.caption_list[:int(0.8*len(self.caption_list))]
        # elif mode == 'valid':
        #     self.image_list = self.image_list[int(0.8*len(self.image_list)):]
        #     self.caption_list = self.caption_list[int(0.8*len(self.caption_list)):]
        # else:
        #     pass   # for mode is 'test'

        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        print(f"Len of Dataset {len(self.image_list)}")

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        # image = os.path.join(self.root_path,'Images',self.image_list[idx].replace('mask_',''))
        # gt = os.path.join(self.root_path,'GTs', self.image_list[idx])
        image_name = self.image_list[idx]
        base_name = os.path.splitext(image_name)[0]

        # KVASIR
        # image = os.path.join(self.root_path, 'images', image_name)
        # gt = os.path.join(self.root_path, 'masks', base_name + ".png")

        # BUSI
        # image_name = self.image_list[idx]

        # image = os.path.join(self.root_path, 'images', image_name)
        # gt = os.path.join(self.root_path, 'masks', image_name)

        if 'mask_name' in self.data.columns:
            image_name = self.data.iloc[idx]['image_name']
            mask_name = self.data.iloc[idx]['mask_name']
        else:
            image_name = self.image_list[idx]
            mask_name = image_name

        image = os.path.join(self.root_path, 'images', image_name)
        gt = os.path.join(self.root_path, 'masks', mask_name)
        # print("GT unique:", torch.unique(gt))
        
        # image = os.path.join(self.root_path,'images', self.image_list[idx])
        # gt = os.path.join(self.root_path,'masks', self.image_list[idx])
        caption = self.caption_list[idx]

        # token_output = self.tokenizer.encode_plus(caption, padding='max_length',
        #                                                 max_length=24, 
        #                                                 truncation=True,
        #                                                 return_attention_mask=True,
        #                                                 return_tensors='pt')
        token_output = self.tokenizer(
                caption,
                padding='max_length',
                max_length=24,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image,gt,token,mask = data['image'],data['gt'],data['token'],data['mask']
        # gt = torch.where(gt==255,1,0)
        gt = (gt > 0).float()
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 
        print("GT unique:", torch.unique(gt))

        return ([image, text], gt)

    def transform(self,image_size=[224,224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),

            ])

        return trans


