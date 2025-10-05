class PGD_Attacker():
  def __init__(self, model, steps, epsilon, loss_func):

    self.model = model
    self.num_steps = steps
    self.epsilon = epsilon
    self.loss_func = loss_func
    self.step_size = 2.5*self.epsilon/self.num_steps


  def adversarial_image(self, input_image, input_label):

    self.model.eval()
    original_image = input_image.clone().detach().to(device)
    adversarial_image = input_image.clone().detach().to(device) + torch.FloatTensor(input_image.shape).uniform_(-self.epsilon, self.epsilon).to(device)

    for i in torch.arange(self.num_steps):
      adversarial_image.requires_grad = True
      output = self.model(adversarial_image)
      loss = self.loss_func(output.squeeze(), input_label)
      self.model.zero_grad()
      loss.backward()
      adversarial_image = adversarial_image.detach() + self.step_size*adversarial_image.grad.sign()
      adversarial_image = original_image + torch.clamp(adversarial_image - original_image, -self.epsilon, self.epsilon)
      adversarial_image = torch.clamp(adversarial_image, 0, 1)
    return adversarial_image.detach()

