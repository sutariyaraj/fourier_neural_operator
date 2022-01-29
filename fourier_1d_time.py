"""
@author: Raj Sutariya
This file is the Fourier Neural Operator for 1D Time problem such as the Heat Equation based on [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import torch
import torch.nn.functional as F
from timeit import default_timer

from matplotlib import pyplot as plt

from utilities3 import *
import wandb
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(11, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


################################################################
# configs
################################################################
TRAIN_PATH = 'data/heat_N1100_T200_r200.mat'
TEST_PATH = 'data/heat_N1100_T200_r200.mat'

ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

modes = 12
width = 20

r = 1
h = int(((421 - 1) / r) + 1)
s = 200

S = 200
T_in = 10
T = 10
step = 1

path = '2_heat_fourier_1d_rnn_N1100_T200_r200' + str(ntrain) + '_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/' + path

# wandb setup
wandb.login()
config = dict(
    learning_rate=learning_rate,
    modes=12,
    width=20,
    epochs=epochs,
    batch_size=batch_size,
    dataset_id="heat_N1100_T200_r200.mat",
)

wandb.init(
    project="fourier_neural_operator",
    notes="tweak baseline",
    tags=["1d_time", "heat_equation"],
    config=config,
)

################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('u')[:ntrain, :, :T_in]
y_train = reader.read_field('u')[:ntrain, :, T_in:T + T_in]

reader.load_file(TEST_PATH)
x_test = reader.read_field('u')[-ntest:, :, :T_in]
y_test = reader.read_field('u')[-ntest:, :, T_in:T + T_in]

print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(ntrain, s, T_in)
x_test = x_test.reshape(ntest, s, T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
myloss = LpLoss(size_average=False)

model = FNO1d(modes, width).cuda()
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    wandb.log({
        "Epoch": ep,
        "Time Taken": t2 - t1,
        "Training Loss per time step": train_l2_step / ntrain / (T / step),
        "Training Loss": train_l2_full / ntrain,
        "Test Loss per time step": test_l2_step / ntest / (T / step),
        "Test Loss": test_l2_full / ntest
    })

torch.save(model, path_model)

################################################################
# prediction on loaded model
################################################################
# model = torch.load(path_model)
# pred = torch.zeros(y_test.shape)
# index = 0
# # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
#
# test_l2_step = 0
# test_l2_full = 0
#
# def show_plot(data_mat):
#     plt.imshow(data_mat, interpolation='nearest', cmap='rainbow',
#                origin='lower', aspect='auto')
#     plt.ylabel('x (cm)')
#     plt.xlabel('t (milliseconds)')
#     plt.axis()
#     plt.colorbar().set_label('Temperature (°C)')
#     plt.show()
# with torch.no_grad():
#     for xx, yy in test_loader:
#         loss = 0
#         xx_original = xx.clone()
#         xx = xx.to(device)
#         yy = yy.to(device)
#
#         for t in range(0, T, step):
#             y = yy[..., t:t + step]
#             im = model(xx)
#             loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
#
#             if t == 0:
#                 pred = im
#             else:
#                 pred = torch.cat((pred, im), -1)
#
#             xx = torch.cat((xx[..., step:], im), dim=-1)
#         for i in range(batch_size):
#             show_plot(torch.cat((xx_original[i], pred[i].cpu()), -1))
#         exit()
#
#         test_l2_step += loss.item()
#         test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
#
# scipy.io.savemat('pred/' + path + '.mat', mdict={'pred': pred.cpu().numpy()})
