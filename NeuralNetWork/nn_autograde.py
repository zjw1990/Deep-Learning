import torch
import  torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sample_size, input_size, hidden_size, output_size = 64, 1000, 100, 10

x = torch.randn(sample_size, input_size, device = device)
y = torch.randn(sample_size, output_size, device = device)

w2 = torch.randn(input_size, hidden_size, device = device, requires_grad = True)
w3 = torch.randn(hidden_size, output_size, device = device, requires_grad = True)

learning_rate = 1e-6
for i in range(500):
    z = x.mm(w2)
    # relu activation
    a = F.relu(z)
    y_pred = a.mm(w3)

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

    