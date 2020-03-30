from loader import *
from network import high_level,low_level
import torchvision.transforms as standard_transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils import str_to_class,measure,save_model,load_model

writer = SummaryWriter(comment = 'code')

torch.manual_seed(999)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(999)
np.random.seed(999)

train_args = {
'lr' : 1e-4,
'epoch' : 100,
'device':'cuda:0',
'model_type' :'hybrid', #high_level,low_level,hybrid
'batch_size': 20,
'num_worker':20,
'train_iter': 0,
}

test_args = {
'batch_size':3498,
'num_worker':20,
'device': 'cuda:0',
'test_iter':0
}

def main(train_args,test_args):

    train_set = OneMillion()
    test_set = OneMillion(is_test = True)
    
    train_loader = DataLoader(train_set,batch_size = train_args['batch_size'],num_workers = train_args['num_worker'],shuffle = True, pin_memory=True)
    test_loader = DataLoader(test_set,batch_size = test_args['batch_size'],num_workers = test_args['num_worker'],shuffle = False, pin_memory=True)
    print('Test set length: {}'.format(test_set.__len__()))
    print('Train set length: {}'.format(train_set.__len__()))

    model = str_to_class(train_args['model_type']).to(train_args['device'])
 
    optimizer = optim.Adam(model.parameters(),lr = train_args['lr'])#optim.SGD(model.parameters(),lr = train_args['lr'])

    criterion = nn.BCEWithLogitsLoss()

    train(model,train_loader,test_loader,optimizer,criterion,train_args,test_args)
    

def train(model,data_train,data_test,optimizer,criterion,train_args,test_args):

    for epoch in tqdm(range(train_args['epoch'])):
        model.train()
        total_loss = 0.0
        number_of_1 = 0
        for _input,_label,_id in data_train:
            _input = _input.to(train_args['device']).float()
            _output = model(_input)

            model.zero_grad()

            _, _label = _label.max(dim = 1)
            _label = _label.unsqueeze(1)
            loss = criterion(_output.to('cpu'),_label.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            total_loss += loss.item()
            train_args['train_iter']+=1
            writer.add_scalar('train_loss',loss.item(),train_args['train_iter'])
        print('epoch {}: training loss {}:'.format(epoch,total_loss/ data_train.__len__()))
        test(model,data_test,optimizer,criterion,test_args,epoch)

 
def test(model,data_test,optimizer,criterion,test_args,epoch):
    model.eval()
    total_loss = 0.0
    ids = None
    f1score = None
    for _input,_label,_id in data_test:
        _input = _input.to(test_args['device']).float()
        _output = torch.sigmoid(model(_input)).to('cpu')
        ids = _id
        _, _label = _label.max(dim = 1)

        precision, recall, f1score,tp,tn,fp,fn = measure(_label,_output)
        writer.add_scalar('precision',precision,test_args['test_iter'])
        writer.add_scalar('recall',recall,test_args['test_iter'])
        writer.add_scalar('f1-score',f1score,test_args['test_iter'])
         
        writer.add_pr_curve('pr-curve',_label,_output.squeeze(),test_args['test_iter'])
        print('precision: {},recall: {},F1-score: {},True Positive: {},True Negative:{},False Positive:{},False Negative:{}'.format(precision,recall,f1score,tp,tn,fp,fn))
        _label = _label.unsqueeze(1)
        loss = criterion(_output.to('cpu'),_label.float())

        total_loss += loss.item()
        writer.add_scalar('test_loss',loss.item(),test_args['test_iter'])
        test_args['test_iter']+=1
    
    print('testing error: {}'.format(total_loss/data_test.__len__()))
    print('-----------------------------------------------')
    save_model(model,_label,_output,ids,epoch,f1score)

main(train_args,test_args)
