import torch
import torch.nn as nn

def main():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    t1 = torch.randn(1,2)
    t2 = torch.randn(1,2).to(dev)

    print('printing both regular tensor (t1) and cuda tensor (t2)')
    print(t1)  # tensor([[-0.2678,  1.9252]])
    print(t2)  # tensor([[ 0.5117, -3.6247]], device='cuda:0')

    t1.to(dev)
    print('t1 = ', t1)  # tensor([[-0.2678,  1.9252]])
    print('is this a cuda tensor?', t1.is_cuda) # False

    t1 = t1.to(dev)
    print('t1 = ', t1)  # tensor([[-0.2678,  1.9252]], device='cuda:0')
    print('is t1 a cuda tensor?', t1.is_cuda) # True

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(1,2)

        def forward(self, x):
            x = self.l1(x)
            return x
    print("testing a simple NN")
    model = M()   # not on cuda
    model.to(dev) # is on cuda (all parameters)
    print("running model on cuda")
    print(next(model.parameters()).is_cuda) # True

if __name__ == '__main__':
    main()

