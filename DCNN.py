import torch 
import torch.nn as nn
import torch.nn.functional as F

class DCNN(nn.Module):
    def __init__(self, H_in, W_in):
        super(DCNN, self).__init__()
        #Conv1: 32 filters, kernel size(3x11)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 11))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 3))
        
        #Conv2: 32 filters, kernel size(2x11)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 11))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 3))
        
        H, W = self._compute_feature_size(H_in, W_in)
        X = H * W 
        
        #FC layers
        self.fc1 = nn.Linear(32 * X, 100)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, 2)
        
    def _compute_feature_size(self, H, W):
        
        H = H - 2  # Conv1 (kernel 3x11, no padding)
        W = W - 10
        H = H // 2  # Pool1 (kernel 2x3)
        W = W // 3

        H = H - 1  # Conv2 (kernel 2x11)
        W = W - 10
        H = H // 2  # Pool2 (kernel 2x3)
        W = W // 3

        return H, W
     
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu2(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
if __name__ == "__main__":
    H_in, W_in = 12, 120
    model = DCNN(H_in, W_in)
    print(model)
    
    # Kiểm tra đầu ra
    sample_input = torch.randn(1, 1, H_in, W_in)
    output = model(sample_input)
    print(output.shape)