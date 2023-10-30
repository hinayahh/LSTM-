from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import json
import glob


class Word_Embedding():
    def __init__(self):
        self.word_dict = {}
        self.tag_dict={}
        self.matrix = []

    def get_words(self):
        for poem in self.data:
            for word in poem:
                if word not in self.word_dict:
                    self.tag_dict[len(self.word_dict)]=word
                    self.word_dict[word] = len(self.word_dict)

    def get_id(self):
        for poem in self.data:
            self.matrix.append([self.word_dict[word] for word in poem])

    def data_load(self):
        self.data = []
        data = []

        # 使用glob获取匹配的文件名列表
        file_list = glob.glob('chinese-poetry-master\Tang-Song\poet.tang.*.json')
        # file_list.extend(glob.glob('chinese-poetry-master\Tang-Song\poet.song.*.json'))

        # 遍历文件列表
        for file_name in file_list:
            with open(file_name, 'r', encoding='UTF-8') as f:
                data = json.load(f)
            
            # 遍历数据中的每个元素，将两个字符串拼接成一个字符串，并添加到poem列表中
            for item in data:
                temp = ''.join(item['paragraphs'])
                if temp == '' or temp == '空。' or temp == '無正文。' or temp == '。' or temp.find('（') != -1 or temp.find('《') != -1\
                    or temp.find('[') != -1 or temp.find('{') != -1:
                    continue
                self.data.append(temp)
        # print(self.data.__len__())
        



    def data_process(self):
        self.data.sort(key=lambda x: len(x))
        self.get_words()
        self.get_id()
        # print(self.matrix)


class ClsDataset(Dataset):
    def __init__(self, poem):
        self.poem = poem

    def __getitem__(self, item):
        return self.poem[item]

    def __len__(self):
        return len(self.poem)


def collate_fn(batch_data):
    poems = batch_data
    poems = [torch.LongTensor([1, *poem]) for poem in poems]

    padded_poems = pad_sequence(poems, batch_first=True, padding_value=0)
    padded_poems = [torch.cat([poem, torch.LongTensor([2])]) for poem in padded_poems]
    padded_poems = list(map(list,padded_poems))
    return torch.LongTensor(padded_poems)


def get_batch(x, batch_size):
    dataset = ClsDataset(x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return dataloader
