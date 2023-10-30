from feature import get_batch, Word_Embedding
from torch import optim
import random, numpy, torch
from Neural_Network import Language
import torch.nn.functional as F
from opencc import OpenCC

seed = random.randint(0,10000)
random.seed(seed)   
numpy.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

dataSet=Word_Embedding()
dataSet.data_load()
dataSet.data_process()

cc = OpenCC('s2t')
time = '22-36-54'

def cat_poem(l):
	"""拼接诗句"""
	poem=list()
	for item in l:
		poem.append(''.join(item))
	return poem

model = Language(50,len(dataSet.word_dict),50,dataSet.tag_dict,dataSet.word_dict) 
model = model.cuda()
model.load_state_dict(torch.load("model\\lstm-" + time + ".pt"))

# 生成随机诗句
# poem=cat_poem(model.generate_random_poem(12,6,random=True))
# for sent in poem:
# 	print(sent)

print("")

# 生成随机藏头诗
torch.manual_seed(seed)
cc = OpenCC('s2t')
poem=cat_poem(model.generate_hidden_head(cc.convert("风林火山"),max_len=7,random=True))
cc = OpenCC('t2s')
for sent in poem:
	print(cc.convert(sent))

