{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Inference and Validation\n",
    "\n",
    "Now that you have a trained network, you can use it for making predictions. This is typically called **inference**, a term borrowed from statistics. However, neural networks have a tendency to perform *too well* on the training data and aren't able to generalize to data that hasn't been seen before. This is called **overfitting** and it impairs inference performance. To test for overfitting while training, we measure the performance on data not in the training set called the **validation** set. We avoid overfitting through regularization such as dropout while monitoring the validation performance during training. In this notebook, I'll show you how to do this in PyTorch. \n",
    "\n",
    "As usual, let's start by loading the dataset through torchvision. You'll learn more about torchvision and loading data in a later part. This time we'll be taking advantage of the test set which you can get by setting `train=False` here:\n",
    "\n",
    "```python\n",
    "testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)\n",
    "```\n",
    "\n",
    "The test set contains images just like the training set. Typically you'll see 10-20% of the original dataset held out for testing and validation with the rest being used for training."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torchvision import datasets, transforms\n",
    "\n",
    "# Define a transform to normalize the data\n",
    "transform = transforms.Compose([transforms.ToTensor(),\n",
    "                                transforms.Normalize((0.5,), (0.5,))])\n",
    "# Download and load the training data\n",
    "trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)\n",
    "trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)\n",
    "\n",
    "# Download and load the test data\n",
    "testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)\n",
    "testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here I'll create a model like normal, using the same one from my solution for part 4."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from torch import nn, optim\n",
    "import torch.nn.functional as F\n",
    "\n",
    "class Classifier(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.fc1 = nn.Linear(784, 256)\n",
    "        self.fc2 = nn.Linear(256, 128)\n",
    "        self.fc3 = nn.Linear(128, 64)\n",
    "        self.fc4 = nn.Linear(64, 10)\n",
    "        \n",
    "    def forward(self, x):\n",
    "        # make sure input tensor is flattened\n",
    "        x = x.view(x.shape[0], -1)\n",
    "        \n",
    "        x = F.relu(self.fc1(x))\n",
    "        x = F.relu(self.fc2(x))\n",
    "        x = F.relu(self.fc3(x))\n",
    "        x = F.log_softmax(self.fc4(x), dim=1)\n",
    "        \n",
    "        return x"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The goal of validation is to measure the model's performance on data that isn't part of the training set. Performance here is up to the developer to define though. Typically this is just accuracy, the percentage of classes the network predicted correctly. Other options are [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)) and top-5 error rate. We'll focus on accuracy here. First I'll do a forward pass with one batch from the test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([64, 10])\n"
     ]
    }
   ],
   "source": [
    "model = Classifier()\n",
    "\n",
    "images, labels = next(iter(testloader))\n",
    "# Get the class probabilities\n",
    "ps = torch.exp(model(images))\n",
    "# Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples\n",
    "print(ps.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "With the probabilities, we can get the most likely class using the `ps.topk` method. This returns the $k$ highest values. Since we just want the most likely class, we can use `ps.topk(1)`. This returns a tuple of the top-$k$ values and the top-$k$ indices. If the highest value is the fifth element, we'll get back 4 as the index."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[4],\n",
      "        [0],\n",
      "        [0],\n",
      "        [4],\n",
      "        [4],\n",
      "        [4],\n",
      "        [0],\n",
      "        [0],\n",
      "        [4],\n",
      "        [4]])\n"
     ]
    }
   ],
   "source": [
    "top_p, top_class = ps.topk(1, dim=1)\n",
    "# Look at the most likely classes for the first 10 examples\n",
    "print(top_class[:10,:])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we can check if the predicted classes match the labels. This is simple to do by equating `top_class` and `labels`, but we have to be careful of the shapes. Here `top_class` is a 2D tensor with shape `(64, 1)` while `labels` is 1D with shape `(64)`. To get the equality to work out the way we want, `top_class` and `labels` must have the same shape.\n",
    "\n",
    "If we do\n",
    "\n",
    "```python\n",
    "equals = top_class == labels\n",
    "```\n",
    "\n",
    "`equals` will have shape `(64, 64)`, try it yourself. What it's doing is comparing the one element in each row of `top_class` with each element in `labels` which returns 64 True/False boolean values for each row."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "equals = top_class == labels.view(*top_class.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we need to calculate the percentage of correct predictions. `equals` has binary values, either 0 or 1. This means that if we just sum up all the values and divide by the number of values, we get the percentage of correct predictions. This is the same operation as taking the mean, so we can get the accuracy with a call to `torch.mean`. If only it was that simple. If you try `torch.mean(equals)`, you'll get an error\n",
    "\n",
    "```\n",
    "RuntimeError: mean is not implemented for type torch.ByteTensor\n",
    "```\n",
    "\n",
    "This happens because `equals` has type `torch.ByteTensor` but `torch.mean` isn't implemented for tensors with that type. So we'll need to convert `equals` to a float tensor. Note that when we take `torch.mean` it returns a scalar tensor, to get the actual value as a float we'll need to do `accuracy.item()`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 10.9375%\n"
     ]
    }
   ],
   "source": [
    "accuracy = torch.mean(equals.type(torch.FloatTensor))\n",
    "print(f'Accuracy: {accuracy.item()*100}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The network is untrained so it's making random guesses and we should see an accuracy around 10%. Now let's train our network and include our validation pass so we can measure how well the network is performing on the test set. Since we're not updating our parameters in the validation pass, we can speed up our code by turning off gradients using `torch.no_grad()`:\n",
    "\n",
    "```python\n",
    "# turn off gradients\n",
    "with torch.no_grad():\n",
    "    # validation pass here\n",
    "    for images, labels in testloader:\n",
    "        ...\n",
    "```\n",
    "\n",
    ">**Exercise:** Implement the validation loop below and print out the total accuracy after the loop. You can largely copy and paste the code from above, but I suggest typing it in because writing it out yourself is essential for building the skill. In general you'll always learn more by typing it rather than copy-pasting. You should be able to get an accuracy above 80%."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1/30..  Training Loss: 0.519..  Test Loss: 0.468..  Test Accuracy: 0.829\n",
      "Epoch: 2/30..  Training Loss: 0.390..  Test Loss: 0.403..  Test Accuracy: 0.853\n",
      "Epoch: 3/30..  Training Loss: 0.353..  Test Loss: 0.378..  Test Accuracy: 0.865\n",
      "Epoch: 4/30..  Training Loss: 0.335..  Test Loss: 0.410..  Test Accuracy: 0.854\n",
      "Epoch: 5/30..  Training Loss: 0.315..  Test Loss: 0.377..  Test Accuracy: 0.869\n",
      "Epoch: 6/30..  Training Loss: 0.304..  Test Loss: 0.378..  Test Accuracy: 0.873\n",
      "Epoch: 7/30..  Training Loss: 0.288..  Test Loss: 0.411..  Test Accuracy: 0.865\n",
      "Epoch: 8/30..  Training Loss: 0.288..  Test Loss: 0.374..  Test Accuracy: 0.873\n",
      "Epoch: 9/30..  Training Loss: 0.273..  Test Loss: 0.357..  Test Accuracy: 0.877\n",
      "Epoch: 10/30..  Training Loss: 0.264..  Test Loss: 0.365..  Test Accuracy: 0.876\n",
      "Epoch: 11/30..  Training Loss: 0.257..  Test Loss: 0.400..  Test Accuracy: 0.873\n",
      "Epoch: 12/30..  Training Loss: 0.255..  Test Loss: 0.379..  Test Accuracy: 0.871\n",
      "Epoch: 13/30..  Training Loss: 0.247..  Test Loss: 0.402..  Test Accuracy: 0.875\n",
      "Epoch: 14/30..  Training Loss: 0.247..  Test Loss: 0.354..  Test Accuracy: 0.879\n",
      "Epoch: 15/30..  Training Loss: 0.238..  Test Loss: 0.355..  Test Accuracy: 0.880\n",
      "Epoch: 16/30..  Training Loss: 0.234..  Test Loss: 0.389..  Test Accuracy: 0.878\n",
      "Epoch: 17/30..  Training Loss: 0.228..  Test Loss: 0.366..  Test Accuracy: 0.882\n",
      "Epoch: 18/30..  Training Loss: 0.223..  Test Loss: 0.391..  Test Accuracy: 0.878\n",
      "Epoch: 19/30..  Training Loss: 0.221..  Test Loss: 0.434..  Test Accuracy: 0.872\n",
      "Epoch: 20/30..  Training Loss: 0.220..  Test Loss: 0.375..  Test Accuracy: 0.877\n",
      "Epoch: 21/30..  Training Loss: 0.213..  Test Loss: 0.396..  Test Accuracy: 0.878\n",
      "Epoch: 22/30..  Training Loss: 0.211..  Test Loss: 0.393..  Test Accuracy: 0.880\n",
      "Epoch: 23/30..  Training Loss: 0.206..  Test Loss: 0.410..  Test Accuracy: 0.880\n",
      "Epoch: 24/30..  Training Loss: 0.206..  Test Loss: 0.409..  Test Accuracy: 0.881\n",
      "Epoch: 25/30..  Training Loss: 0.199..  Test Loss: 0.420..  Test Accuracy: 0.881\n",
      "Epoch: 26/30..  Training Loss: 0.196..  Test Loss: 0.424..  Test Accuracy: 0.882\n",
      "Epoch: 27/30..  Training Loss: 0.202..  Test Loss: 0.405..  Test Accuracy: 0.886\n",
      "Epoch: 28/30..  Training Loss: 0.184..  Test Loss: 0.433..  Test Accuracy: 0.880\n",
      "Epoch: 29/30..  Training Loss: 0.192..  Test Loss: 0.404..  Test Accuracy: 0.882\n",
      "Epoch: 30/30..  Training Loss: 0.191..  Test Loss: 0.394..  Test Accuracy: 0.883\n"
     ]
    }
   ],
   "source": [
    "model = Classifier()\n",
    "criterion = nn.NLLLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=0.003)\n",
    "\n",
    "epochs = 30\n",
    "steps = 0\n",
    "\n",
    "train_losses, test_losses = [], []\n",
    "for e in range(epochs):\n",
    "    running_loss = 0\n",
    "    for images, labels in trainloader:\n",
    "        \n",
    "        optimizer.zero_grad()\n",
    "        \n",
    "        log_ps = model(images)\n",
    "        loss = criterion(log_ps, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "        running_loss += loss.item()\n",
    "        \n",
    "    else:\n",
    "        test_loss = 0\n",
    "        accuracy = 0\n",
    "\n",
    "        # Turn off gradients for validation, saves memory and computations\n",
    "        with torch.no_grad():\n",
    "            for images, labels in testloader:\n",
    "                log_ps = model(images) # this is log of the probablilities\n",
    "                test_loss += criterion(log_ps, labels)\n",
    "\n",
    "                ps = torch.exp(log_ps) #probability distribution\n",
    "                top_p, top_class = ps.topk(1, dim=1)\n",
    "                equals = top_class == labels.view(*top_class.shape)\n",
    "                accuracy += torch.mean(equals.type(torch.FloatTensor))\n",
    "\n",
    "        train_losses.append(running_loss/len(trainloader))\n",
    "        test_losses.append(test_loss/len(testloader))\n",
    "\n",
    "        print(\"Epoch: {}/{}.. \".format(e+1, epochs),\n",
    "              \"Training Loss: {:.3f}.. \".format(running_loss/len(trainloader)),\n",
    "              \"Test Loss: {:.3f}.. \".format(test_loss/len(testloader)),\n",
    "              \"Test Accuracy: {:.3f}\".format(accuracy/len(testloader)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Overfitting\n",
    "\n",
    "If we look at the training and validation losses as we train the network, we can see a phenomenon known as overfitting.\n",
    "\n",
    "<img src='assets/overfitting.png' width=450px>\n",
    "\n",
    "The network learns the training set better and better, resulting in lower training losses. However, it starts having problems generalizing to data outside the training set leading to the validation loss increasing. The ultimate goal of any deep learning model is to make predictions on new data, so we should strive to get the lowest validation loss possible. One option is to use the version of the model with the lowest validation loss, here the one around 8-10 training epochs. This strategy is called *early-stopping*. In practice, you'd save the model frequently as you're training then later choose the model with the lowest validation loss.\n",
    "\n",
    "The most common method to reduce overfitting (outside of early-stopping) is *dropout*, where we randomly drop input units. This forces the network to share information between weights, increasing it's ability to generalize to new data. Adding dropout in PyTorch is straightforward using the [`nn.Dropout`](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout) module.\n",
    "\n",
    "```python\n",
    "class Classifier(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.fc1 = nn.Linear(784, 256)\n",
    "        self.fc2 = nn.Linear(256, 128)\n",
    "        self.fc3 = nn.Linear(128, 64)\n",
    "        self.fc4 = nn.Linear(64, 10)\n",
    "        \n",
    "        # Dropout module with 0.2 drop probability\n",
    "        self.dropout = nn.Dropout(p=0.2)\n",
    "        \n",
    "    def forward(self, x):\n",
    "        # make sure input tensor is flattened\n",
    "        x = x.view(x.shape[0], -1)\n",
    "        \n",
    "        # Now with dropout\n",
    "        x = self.dropout(F.relu(self.fc1(x)))\n",
    "        x = self.dropout(F.relu(self.fc2(x)))\n",
    "        x = self.dropout(F.relu(self.fc3(x)))\n",
    "        \n",
    "        # output so no dropout here\n",
    "        x = F.log_softmax(self.fc4(x), dim=1)\n",
    "        \n",
    "        return x\n",
    "```\n",
    "\n",
    "During training we want to use dropout to prevent overfitting, but during inference we want to use the entire network. So, we need to turn off dropout during validation, testing, and whenever we're using the network to make predictions. To do this, you use `model.eval()`. This sets the model to evaluation mode where the dropout probability is 0. You can turn dropout back on by setting the model to train mode with `model.train()`. In general, the pattern for the validation loop will look like this, where you turn off gradients, set the model to evaluation mode, calculate the validation loss and metric, then set the model back to train mode.\n",
    "\n",
    "```python\n",
    "# turn off gradients\n",
    "with torch.no_grad():\n",
    "    \n",
    "    # set model to evaluation mode\n",
    "    model.eval()\n",
    "    \n",
    "    # validation pass here\n",
    "    for images, labels in testloader:\n",
    "        ...\n",
    "\n",
    "# set model back to train mode\n",
    "model.train()\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> **Exercise:** Add dropout to your model and train it on Fashion-MNIST again. See if you can get a lower validation loss or higher accuracy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## TODO: Define your model with dropout added\n",
    "class Classifier(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.fc1 = nn.Linear(784, 256)\n",
    "        self.fc2 = nn.Linear(256, 128)\n",
    "        self.fc3 = nn.Linear(128, 64)\n",
    "        self.fc4 = nn.Linear(64, 10)\n",
    "\n",
    "        # Dropout module with 0.2 drop probability\n",
    "        self.dropout = nn.Dropout(p=0.2)\n",
    "    def forward(self, x):\n",
    "        # make sure input tensor is flattened\n",
    "        x = x.view(x.shape[0], -1)\n",
    "\n",
    "        # Now with dropout\n",
    "        x = self.dropout(F.relu(self.fc1(x)))\n",
    "        x = self.dropout(F.relu(self.fc2(x)))\n",
    "        x = self.dropout(F.relu(self.fc3(x)))\n",
    "\n",
    "        # output so no dropout here\n",
    "        x = F.log_softmax(self.fc4(x), dim=1)\n",
    "\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## TODO: Train your model with dropout, and monitor the training progress with the validation loss and accuracy\n",
    "model = Classifier()\n",
    "criterion = nn.NLLLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=0.003)\n",
    "\n",
    "epochs = 30\n",
    "steps = 0\n",
    "\n",
    "train_losses, test_losses = [], []\n",
    "for e in range(epochs):\n",
    "    running_loss = 0\n",
    "    for images, labels in trainloader:\n",
    "        \n",
    "        optimizer.zero_grad()\n",
    "        \n",
    "        log_ps = model(images)\n",
    "        loss = criterion(log_ps, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "        running_loss += loss.item()\n",
    "        \n",
    "    else:\n",
    "        test_loss = 0\n",
    "        accuracy = 0\n",
    "\n",
    "        # Turn off gradients for validation, saves memory and computations\n",
    "        with torch.no_grad():\n",
    "            for images, labels in testloader:\n",
    "                log_ps = model(images) # this is log of the probablilities\n",
    "                test_loss += criterion(log_ps, labels)\n",
    "\n",
    "                ps = torch.exp(log_ps) #probability distribution\n",
    "                top_p, top_class = ps.topk(1, dim=1)\n",
    "                equals = top_class == labels.view(*top_class.shape)\n",
    "                accuracy += torch.mean(equals.type(torch.FloatTensor))\n",
    "        \n",
    "        model.train()\n",
    "        \n",
    "        train_losses.append(running_loss/len(trainloader))\n",
    "        test_losses.append(test_loss/len(testloader))\n",
    "\n",
    "        print(\"Epoch: {}/{}.. \".format(e+1, epochs),\n",
    "              \"Training Loss: {:.3f}.. \".format(running_loss/len(trainloader)),\n",
    "              \"Test Loss: {:.3f}.. \".format(test_loss/len(testloader)),\n",
    "              \"Test Accuracy: {:.3f}\".format(accuracy/len(testloader)))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Inference\n",
    "\n",
    "Now that the model is trained, we can use it for inference. We've done this before, but now we need to remember to set the model in inference mode with `model.eval()`. You'll also want to turn off autograd with the `torch.no_grad()` context."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import helper module (should be in the repo)\n",
    "import helper\n",
    "\n",
    "# Test out your network!\n",
    "\n",
    "model.eval()\n",
    "\n",
    "dataiter = iter(testloader)\n",
    "images, labels = dataiter.next()\n",
    "img = images[0]\n",
    "# Convert 2D image to 1D vector\n",
    "img = img.view(1, 784)\n",
    "\n",
    "# Calculate the class probabilities (softmax) for img\n",
    "with torch.no_grad():\n",
    "    output = model.forward(img)\n",
    "\n",
    "ps = torch.exp(output)\n",
    "\n",
    "# Plot the image and probabilities\n",
    "helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Next Up!\n",
    "\n",
    "In the next part, I'll show you how to save your trained models. In general, you won't want to train a model everytime you need it. Instead, you'll train once, save it, then load the model when you want to train more or use if for inference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
