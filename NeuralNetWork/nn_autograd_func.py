import torch
import torch.nn.functional as F 
class MyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.relu(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x<0] = 0
        return grad_x



dtype = torch.float 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sample_size, input_size, hidden_size, output_size = 64, 1000, 100, 10

x = torch.randn(sample_size, input_size, device = device, dtype=dtype)
y = torch.randn(sample_size, output_size, device = device, dtype = dtype)

w2 = torch.randn(input_size, hidden_size, device = device, requires_grad = True, dtype = dtype)
w3 = torch.randn(hidden_size, output_size, device = device, requires_grad = True, dtype = dtype)
learning_rate = 1e-6


for i in range(500):
    relu = MyRelu.apply

    y_pred = relu(x.mm(w2)).mm(w3)

    # back propagation
    loss = (y_pred - y).pow(2).sum()
    print(i, loss.item())
    loss.backward()
    # Update weights
    with torch.no_grad():
        w2 -= learning_rate * w2.grad
        w3 -= learning_rate * w3.grad

        w2.grad.zero_()
        w3.grad.zero_()