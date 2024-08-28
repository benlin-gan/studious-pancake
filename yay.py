from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import data

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
base_model = AutoModel.from_pretrained("gpt2-large")
# gpt-2 uses deembeding equal to transpose of embedding
# hence, 'yes' (8505) - 'no' (3919) gives a logit for predicted class membership
with torch.no_grad():
    default_probe = (base_model.wte.weight[8505] - base_model.wte.weight[3919])

class Probe(torch.nn.Module): 
    def __init__(self, base_model):
        super(Probe, self).__init__()
        self.base_model = base_model
        self.initial = nn.Parameter(default_probe)
        self.trainable = nn.Parameter(torch.zeros_like(self.initial))
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.trainable.requires_grad = True

    def forward(self, x):
        print("start llm")
        final_acts = self.base_model(x).last_hidden_state[:, -1, :] #get final position vector
        print("end llm")
        self.unmodified_logit = final_acts @ self.initial
        return self.unmodified_logit + final_acts @ self.trainable
        

def frappe_mmd_loss(probe_model, batch, label, sensitive, lambd=0.2):
    """
        mmd for fairness was invented as an in-processing technique
        FRAPPE allows us to use it for post-processing 
    """
    output = probe_model(batch)
    p_tuned = F.sigmoid(output)
    p_base = F.sigmoid(probe_model.unmodified_logit)
    kl_loss = p_base * torch.log(p_base/p_tuned) + (1 - p_base) * torch.log((1 - p_base)/(1 - p_tuned)) # don't move too far from base model distribution (FRAPPE trick)

    fpns = output[~label & ~sensitive] #scores for majority group (with negative label)
    fps = output[~label & sensitive] #scores for minority group (with negative label) 
    
    N = min(fpns.shape[0], fps.shape[0])
    mmd_loss = mmd(fps[:N, :], fpns[:N, :]) #push distributions of fps and fpns together (equal opportunity - same outputs for both groups for one value of true label)
    print("kl_loss=" + kl_loss, "mmd_loss=" + mmd_loss * lambd)
    return kl_loss + lambd * mmd_loss

def mmd(x, y, alpha=-0.5):
    """
        x - (N, D)
        y - (N, D)
        https://discuss.pytorch.org/t/maximum-mean-discrepancy-mmd-and-radial-basis-function-rbf/1875
        match distribution x and y 
    """
    B = x.shape[0]
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = torch.exp(-alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(-alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(-alpha * (rx.t() + ry - 2*zz))

    beta = (1./(B*(B-1)))
    gamma = (2./(B*B)) 
    print(K, L, P)

    return beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)

def main():
    probe_model = Probe(base_model)
    batch_generator = data.gen_batch('datasets/toxic_comments.csv', 'muslim')
    num_epochs = 10
    optimizer = torch.optim.SGD(probe_model.parameters(), lr=0.001, momentum=0.9)
    for e in range(num_epochs):
        print(f"epoch {e}")
        running_loss = 0.0
        for batch_idx, (features, toxic, sensitive) in enumerate(batch_generator):
            #forward pass happens inside frappe_mmd_loss
            loss = frappe_mmd_loss(probe_model, features, toxic, sensitive)
            loss.backward()
            optimizer.step()
            if i % 200 == 199:
                print('[%d, %5D] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

if __name__ == '__main__' : 
    main()
