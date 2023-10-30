from feature import get_batch, Word_Embedding
from torch import optim
import random, numpy, torch
from Neural_Network import Language
import torch.nn.functional as F
from tqdm import tqdm
import datetime

seed = random.randint(0,10000)

random.seed(seed)   
numpy.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

dataSet=Word_Embedding()
dataSet.data_load()
dataSet.data_process()
train=get_batch(dataSet.matrix,1)
learning_rate=0.004
iter_times=1


train_loss_records=list()

model=Language(50, len(dataSet.word_dict), 50, dataSet.tag_dict, dataSet.word_dict)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fun = F.cross_entropy
train_loss_record = []

model=model.cuda()
for iteration in range(iter_times):
	print("---------- Iteration", iteration + 1, "----------")
	total_loss = 0
	model.train()
	loop = tqdm(enumerate(train), total=len(train))
	for i, batch in loop:
		x=batch.cuda()
		x,y=x[:,:-1],x[:,1:]
		pred = model(x).transpose(1,2)
		optimizer.zero_grad()
		loss = loss_fun(pred, y)
		total_loss+=loss.item()/(x.shape[1]-1)
		loss.backward()
		optimizer.step()
	train_loss_record.append(total_loss/len(train))
	print("Train loss:", total_loss/len(train))

	time = str(datetime.datetime.now().strftime("%H-%M-%S"))

	torch.save(model.state_dict(), "model\\lstm-" + time + ".pt")
