
futures = []
for i in range(0, 100):
  future = client.submit(gen_data, 5, 16, i)
  futures.append(future)
results = client.gather(futures)

# Training size: 10M
# Validation size: 5M
  
# Paths: 8.192M
# Standard Deviation: 0.0073

# Activation Function: Elu (replace ReLu)

import torch.nn as nn
import torch.nn.functional as F
import torch
 
 
class Net(nn.Module):
 
    def __init__(self, hidden=1024):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, hidden)
        self.fc6 = nn.Linear(hidden, 1)
        self.register_buffer('norm',
                             torch.tensor([200.0,
                                           198.0,
                                           200.0,
                                           0.4,
                                           0.2,
                                           0.2]))
 
    def forward(self, x):
        # normalize the parameter to range [0-1] 
        x = x / self.norm
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = F.elu(self.fc5(x))
        return self.fc6(x)

#Inference, Greeks, and implied volatility calculation

checkpoint = torch.load('check_points/512/model_best.pth.tar')
model = Net().cuda()
model.load_state_dict(checkpoint['state_dict'])
inputs = torch.tensor([[110.0, 100.0, 120.0, 0.35, 0.1, 0.05]])
start = time.time()
inputs = inputs.cuda()
result = model(inputs)
end = time.time()
print('result %.4f inference time %.6f' % (result,end- start)) 

inputs = torch.tensor([[110.0, 100.0, 120.0, 0.35, 0.1, 0.05]]).cuda()
inputs.requires_grad = True
x = model(inputs)
x.backward()
first_order_gradient = inputs.grad 

from torch import Tensor
from torch.autograd import Variable
from torch.autograd import grad
from torch import nn
 
inputs = torch.tensor([[110.0, 100.0, 120.0, 0.35, 0.1, 0.05]]).cuda()
inputs.requires_grad = True
x = model(inputs)
 
# instead of using loss.backward(), use torch.autograd.grad() to compute gradients
# https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
loss_grads = grad(x, inputs, create_graph=True)
drv = grad(loss_grads[0][0][2], inputs)
drv

def compute_price(sigma):
    inputs = torch.tensor([[110.0, 100.0, 120.0, sigma, 0.1, 0.05]]).cuda()
    x = model(inputs)
    return x.item()


# Deploy model to production, TensorRT inference speed-up

import tensorrt as trt
import time
import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
 
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
 
with open("opt.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
 
h_input = cuda.pagelocked_empty((1,6,1,1), dtype=np.float32)
h_input[0, 0, 0, 0] = 110.0
h_input[0, 1, 0, 0] = 100.0
h_input[0, 2, 0, 0] = 120.0
h_input[0, 3, 0, 0] = 0.35
h_input[0, 4, 0, 0] = 0.1
h_input[0, 5, 0, 0] = 0.05
h_output = cuda.pagelocked_empty((1,1,1,1), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()
with engine.create_execution_context() as context:
    start = time.time()
    cuda.memcpy_htod_async(d_input, h_input, stream)
    input_shape = (1, 6, 1, 1)
    context.set_binding_shape(0, input_shape)
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    end = time.time()
print('result %.4f inference time %.6f' % (h_output,end- start))












