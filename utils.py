import importlib
import torch
from numpy import savetxt


def str_to_class(class_name,module_name = 'network'):
    try:
        module_ = importlib.import_module(module_name)
        try:
            class_ = getattr(module_, class_name)()
        except AttributeError:
            logging.error('Class does not exist')
    except ImportError:
        logging.error('Module does not exist')
    return class_ or None


def measure(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    y_pred = y_pred.squeeze()
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)


    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2* (precision*recall) / (precision + recall + epsilon)
#    f1.requires_grad = is_training

    return precision,recall,f1,tp,tn,fp,fn

def save_model(model,label,output,ids,epoch_num,f1score):
    save_dir = 'checkpoints/' + str(epoch_num) + '_' + str(f1score.data.numpy()) + '.pth'
    state = {'epoch': epoch_num,
             'model': model.state_dict(),
            }
    torch.save(state,save_dir)
    savetxt('checkpoints/data/'+str(epoch_num) +'_'+ 'labels.csv', label.data.numpy(), delimiter=',')
    savetxt('checkpoints/data/'+str(epoch_num) +'_'+ 'outputs.csv', output.data.numpy(), delimiter=',')
    savetxt('checkpoints/data/'+str(epoch_num) +'_'+ 'ids.csv', ids, delimiter=',',fmt='%s')

def load_model(addr,model = None):
    state = torch.load(addr)
    epoch = state['epoch']
    labels = loadtxt('checkpoints/data/'+str(epoch)+'_'+'labels.csv', delimiter=',')
    outputs = loadtxt('checkpoints/data/'+str(epoch)+'_'+'outputs.csv', delimiter=',')
    ids = loadtxt('checkpoints/data/'+str(epoch)+'_'+'ids.csv', delimiter=',')

    if model != None:
        model.load_state_dict(state['model'])
    return epoch,labels,outputs,ids

