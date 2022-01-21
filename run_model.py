import torch as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

import models as nns
import samples as samps
from utils import comp_time,get_time

def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    ts.manual_seed(seed)
    ts.cuda.manual_seed_all(seed)
    # For atomic operations there is currently 
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    ts.backends.cudnn.deterministic = True
    ts.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)

enforce_reproducibility()

device = ts.device("cpu")
if ts.cuda.is_available():
    print("has cuda")
    device = ts.device("cuda")

"""
Converting things into tensors
Converting things into tensors
Converting things into tensors
"""

def x2tensor(x_in):
    x = [float(x) for x in x_in]
    return ts.tensor(x).to(device)

def y2tensor(y):
    return ts.LongTensor([y]).to(device)

"""
Training the model
Training the model
Training the model
"""

def train(model,lrate,n_epochs,optim_fun,D_train,D_test):
    model.to(device)

    N = D_train.shape[0]
    est_time_n = 0

    if n_epochs <= 0:
        return True

    loss_fun = nn.NLLLoss()
    optimizer = optim_fun(model.parameters(),lr=lrate)

    for epoch in range(n_epochs):
        avg_loss = 0
        start_time = get_time()
        for d in D:
            model.zero_grad()

            x = d[:-1]
            y = d[-1]

            xt = x2tensor(x)

            target = y2tensor(y)

            logits = model(xt)

            loss = loss_fun(logits,target)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            est_time_n += 1
            if est_time_n == 1000:
                print("est-time per epoch:" + comp_time(start_time,lambda t0: t0 * N / 1000))
                print("")
                break
                

        
        print("loss[" + str(epoch) + "] = " + str(avg_loss / N))
        print_test(model,D_test)
        print("------------------------------------took " + comp_time(start_time,None))

"""
Testing the model
Testing the model
Testing the model
"""

def run_model(model,sample):
    logits = None
    with ts.no_grad():
        logits = model(x2tensor(sample))
    return logits.argmax().item()

def test(model,D_test):
    model.to(device)

    correct = 0
    misses = 0

    N = D_test.shape[0]

    for d in D_test:
        x = d[:-1]
        y = d[-1]
        pred = run_model(model,x)
        if pred == y:
            correct += 1
        else:
            misses += 1

    return correct,misses,N

def print_test(model,dset_test):
    correct,misses,N_test = test(model,D_test)
    print("---- test run:")
    print("  set size : " + str(N_test))
    print("  correct  : " + str(correct))
    print("  misses   : " + str(misses))
    print("  acc      : " + str(correct / N_test))

def print_pred(x,smap,model):
    model.to(device)
    y = run_model(model,x)
    print(samps.id2m(y,smap))

if __name__ == "__main__":
    model_name = "lrotations_nn1"
    model_file_name = "ptmodel_" + model_name + ".ptm"

    do_test = False
    do_teston = False
    do_train = True
    do_pred_stuff = False

    do_load_model = False

    n_epochs = 1
    l_rate = 1.0 / (10 ** 9)

    D,n,smap = samps.load_dset()

    n_train = round(0.8 * n)

    D_train = D[:n_train]
    D_test  = D[n_train:]


    print("train set size  : " + str(D_train.shape[0]))
    print("test set size   : " + str(D_test.shape[0]))

    model = nns.LearningRotationNN(4 ** 2,200,n)

    if do_load_model:
        print("-- model loaded")
        model.load_state_dict(ts.load(model_file_name))
    
    if do_train: 
        train(model,l_rate,n_epochs,optim.Adam,D_train,D_test)
        ts.save(model.state_dict(),model_file_name)

    if do_test:
        print_test(model,D_test)

    if do_teston:
        print_test(model,D_teston)

    if do_pred_stuff:
        pred_stuff(model)
