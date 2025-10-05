
class Block(nn.Module):
  def __init__(self, input_channels, output_channels, stride, dropout = 0.0):
    super().__init__()
    self.bn1 = nn.BatchNorm2d(input_channels)
    self.bn2 = nn.BatchNorm2d(output_channels)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
    self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
    self.dropout = nn.Dropout(p=dropout)
    self.skip_connection = nn.Sequential()

    if stride != 1 or input_channels != output_channels:  # if the input channels dont match output channels , we just apply
    # a convolution layer to make them same
      self.skip_connection = nn.Conv2d(input_channels, output_channels, kernel_size = 1, stride = stride, bias = False)

  def forward(self, x):
    out = self.conv1(self.relu1(self.bn1(x)))
    out = self.conv2(self.relu2(self.bn2(self.dropout(out))))

    out += self.skip_connection(x)
    return(out)

""" we define the depth to be the total number of layers, so in 28 depth , we have 4 layer outside
 the residual blocks , intial conv, bn, relu, fc, thus for 3 groups/stages there are 4 residual blocks in
 each block"""

""" the channels_list contains 4 elements each specifying the number of out_channels of the group , the first element
is for the initial conv layer"""

class WideResnet(nn.Module):
  def __init__(self, depth, widening_factor, num_classes, dropout = 0.0, channels_list = [16, 16, 32, 64]):
    super().__init__()
    n = (depth - 4)//6
    self.conv1 = nn.Conv2d(3,channels_list[0] , kernel_size = 3, stride = 1, padding = 1, bias = False)
    self.group1 = self.block_stacker(Block, channels_list[0], channels_list[1]*widening_factor, n, 1, dropout)
    self.group2 = self.block_stacker(Block, channels_list[1]*widening_factor, channels_list[2]*widening_factor, n, 2, dropout)
    self.group3 = self.block_stacker(Block, channels_list[2]*widening_factor, channels_list[3]*widening_factor, n, 2, dropout)
    self.bn3 = nn.BatchNorm2d(channels_list[3]*widening_factor)
    self.relu3 = nn.ReLU()
    self.fc = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_classes))
    self.pool = nn.AdaptiveAvgPool2d((1, 1))

  def forward(self, x):
    out = self.conv1(x)
    out = self.group1(out)
    out = self.group2(out)
    out = self.group3(out)
    out = self.relu3(self.bn3(out))
    out = self.pool(out)
    return self.fc(out)

  def block_stacker(self, builder_block, input_channels, output_channels, n, stride, dropout):
    stack = []
    stack.append(builder_block(input_channels, output_channels, stride, dropout))
    for i in torch.arange(1, n):
      stack.append(builder_block(output_channels, output_channels, 1, dropout))
    return nn.Sequential(*stack)
