class cka_pipeline():
  def __init__(self, model, test_dataloader, num_samples, device, batch_size ):
    self.model = model
    self.test_dataloader = test_dataloader
    self.num_samples = num_samples
    self.hooks = []
    self.batch_size = batch_size
    self.device = device
    self.activation_dict = {}
    self.model.eval()
    self.centering_matrix = torch.eye(self.batch_size) - (1/self.batch_size)*torch.ones((self.batch_size, self.batch_size))
    self.centering_matrix = self.centering_matrix.to(self.device)
    self.layer_name = []
    
    for name, layer in self.model.named_modules():
      if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.ReLU, nn.LazyLinear)):
        hook = layer.register_forward_hook(self.hook_func(name))
        self.hooks.append(hook)
        self.layer_name.append(name)

    counter = 0
    self.num_layers = len(self.layer_name)
    print('num layers are ', self.num_layers)
    self.den1 = torch.zeros(self.num_layers)
    self.den2 = torch.zeros(self.num_layers)
    self.num = torch.zeros((self.num_layers, self.num_layers))
    with torch.no_grad():
      for image, label in self.test_dataloader:
        if counter > self.num_samples:
          break
        counter += self.batch_size
        print('images processed', counter)
        image = image.to(self.device)
        label = label.to(self.device)
        _ = self.model(image)
        l = [i for i in self.activation_dict.values()]
        for i in range(self.num_layers):
          print(f"layer number {i}")
          layer = l[i].to(self.device)
          layer = layer@layer.T
          gram = self.centering_matrix@layer@self.centering_matrix
          m = torch.sum(gram*gram).cpu()
          self.den1[i] += m
          self.den2[i] += m
          del layer
        
          for j in range(i , self.num_layers):
            if i == j:
              self.num[i, j] += m
            else:
              layer2 = l[j].to(self.device)
              layer2 = layer2@layer2.T
              gram2 = self.centering_matrix@layer2@self.centering_matrix
              
              del layer2
              m2 = torch.sum(gram*gram2).cpu()
              self.num[i, j] += m2
              self.num[j, i] += m2
              del gram2
          del gram
        self.activation_dict.clear()
        torch.cuda.empty_cache()

      self.similarity_matrix = torch.zeros((self.num_layers, self.num_layers))
      print("populating similarity matrix")
      for i in range(self.num_layers):
        for j in range(self.num_layers):
          d = self.num[i, j]/(torch.sqrt(self.den1[i])*torch.sqrt(self.den2[j]))
          self.similarity_matrix[i, j] = d
          self.similarity_matrix[j, i] = d


  def hook_func(self, name):
    def hook(module, input, output):
      self.activation_dict[name] = output.view(output.size(0), -1).detach().cpu()
    return hook
  
  def remove_hooks(self):
    for hook in self.hooks:
        hook.remove()
