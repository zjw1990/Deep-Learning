import torch
import  torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sample_size, input_size, hidden_size, output_size = 64, 1000, 100, 10

x = torch.randn(sample_size, input_size, device = device)
y = torch.randn(sample_size, output_size, device = device)

w2 = torch.randn(input_size, hidden_size, device = device)
w3 = torch.randn(hidden_size, output_size, device = device)

learning_rate = 1e-6
for i in range(10):
    z = x.mm(w2)
    # relu activation
    a = F.relu(z)
    y_pred = a.mm(w3)

    # back propagation
    loss = (y_pred - y).pow(2).sum()
    print(i, loss.item())

    sigma_3 = 2 * (y_pred - y)
    w_3_grad = a.t().mm(sigma_3)

    sigma_2 = w3.mm(sigma_3.t())
    sigma_2[z.t() < 0] = 0

    w_2_grad = x.t().mm(sigma_2.t())

    # Update weights
    w2 -= learning_rate * w_2_grad
    w3 -= learning_rate * w_3_grad

    
