import pandas as pd
from torch.utils.data import Dataset

class ThuCNewsdataset(Dataset):
    def __init__(
        self,
        data_path,
    ):
        self.data_path = data_path
        self._get_data()
        
    def _get_data(self):
        with open(self.data_path,'r',encoding='utf-8') as f:
            data = f.readlines()
        split_data = []
        for item in data[0:100]:
            split_item = item.split('\t')
            split_data.append(split_item)
        data_split = pd.DataFrame(split_data,columns=('category','content'))
        categories = data_split['category'].unique()
        category_to_code = {category: code for code, category in enumerate(categories)}
        data_split['category'] = data_split['category'].map(category_to_code)
        self.data = data_split
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.loc[idx,'category'], self.data.loc[idx,'content']
        