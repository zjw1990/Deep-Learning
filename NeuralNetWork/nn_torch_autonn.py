import torch
import torch.nn.functional as F
# load data
sample_size, input_size, hidden_size, output_size = 64, 1000, 100, 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x = torch.randn(sample_size, input_size, device = device)
y = torch.randn(sample_size, output_size, device = device)
# define model
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, output_size),
).to(device = device)
# define loss function
loss_function = torch.nn.MSELoss(reduction='sum')
# define optimizer
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training process:
for i in range(500):

    y_pred = model(x)
    loss = loss_function(y_pred, y)
    print(i, loss.item())   
    # zero all gradients
    optimizer.zero_grad()
    # backprop
    loss.backward()
    # update parameters
    optimizer.step()

