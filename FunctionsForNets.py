def check_accuracy(model, dataLoader, device=torch.device('cpu')):
  """
  Checking accuracy on given dataSet in dataLoader
  """
  model = model.to(device=device)

  typeOfCheck = str.lower(typeOfCheck)

  model.eval()

  num_samples = 0
  num_correct = 0

  with torch.no_grad():
    for i, (x, y) in enumerate(dataLoader):
      x = x.to(device=device, dtype=torch.float32)
      y = y.to(device=device, dtype=torch.long)

      scores = model(x)
      _, preds = scores.max(1)

      num_samples += preds.shape[0]
      num_correct += (preds == y).sum()

      #if typeOfCheck == 'train' and i == 100: # Чтобы проверяться не на всём огромном train сете (16 * 64 = 1024 как и у validation)
      #  break

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f) in %s'% (num_correct, num_samples, 100 * acc, typeOfCheck))
    return acc



def train_model(model, optimizer, trainLoader, validationLoader, num_epoch=1, device=torch.device('cpu'), printAndSaveEvery=100, continueTraining=None, savePath = '/content/drive/MyDrive/'):
  """
  Function for making model training

  continueTraining is a dictionary including:
  1) train_accs - list of training accuracies
  2) val_accs - list of validation accuracies
  3) num_ep - number of epoch from which the model will continue training
  """
  import torch.nn.functional as F
  from os.path import join

  if continueTraining is None:
    print("Starting new training")
    best_acc = 0
    train_accuracies = []
    val_accuracies = []
    start_ep = 0
  else:
    train_accuracies = continueTraining['train_accs']
    val_accuracies = continueTraining['val_accs']
    start_ep = continueTraining['num_ep']
    best_acc = val_accuracies[-1]
    print("Continue training from " + str(start_ep) + " epoch")

  model = model.to(device=device)

  for e in range(start_ep, num_epoch):
    print("Start " + str(e) + " epoch")
    for t, (x, y) in enumerate(trainLoader):

      model.train()
      x = x.to(device=device, dtype=torch.float32)
      y = y.to(device=device, dtype=torch.long)

      scores = model(x)
      loss = F.cross_entropy(scores, y)

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

      if t % printAndSaveEvery == 0 and t != 0:
        print("Iteration " + str(t) + ":")
        train_acc = check_accuracy(model, trainLoader, 'train', device)
        val_acc = check_accuracy(model, validationLoader, 'validation', device)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if best_acc < val_acc:
          print("Goten new best val accuracy. Save new best model")
          best_acc = val_acc
          torch.save({
              'Model_state_dict': model.state_dict(),
              'Optimizer_state_dict': optimizer.state_dict(),
              'Num_epoch': e,
              'Train_accs': train_accuracies,
              'Val_accs': val_accuracies
          }, join(savePath, 'best_model.pt'))
          

    train_acc = check_accuracy(model, trainLoader, 'train', device)
    val_acc = check_accuracy(model, validationLoader, 'validation', device)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    torch.save({
        'Model_state_dict': model.state_dict(),
        'Optimizer_state_dict': optimizer.state_dict(),
        'Num_epoch': e + 1,
        'Train_accs': train_accuracies,
        'Val_accs': val_accuracies
    }, join(savePath, 'best_model.pt'))

    if best_acc < val_acc:
          print("Goten new best val accuracy. Save new best model")
          best_acc = val_acc
          torch.save({
              'Model_state_dict': model.state_dict(),
              'Optimizer_state_dict': optimizer.state_dict(),
              'Num_epoch': e,
              'Train_accs': train_accuracies,
              'Val_accs': val_accuracies
          }, join(savePath, 'best_model.pt'))

  return(train_accuracies, val_accuracies)


def show_accuracy_history(train_acc, val_acc):

    if train_acc is not list or val_acc is not list:
        print('train and validation accuracies must be lists')
        return

    plt.plot(range(len(train_acc)), train_acc, range(len(val_acc)), val_acc)
    plt.legend(['train', 'validation'])