{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Transfer Learning\n",
    "\n",
    "In this notebook, you'll learn how to use pre-trained networks to solved challenging problems in computer vision. Specifically, you'll use networks trained on [ImageNet](http://www.image-net.org/) [available from torchvision](http://pytorch.org/docs/0.3.0/torchvision/models.html). \n",
    "\n",
    "ImageNet is a massive dataset with over 1 million labeled images in 1000 categories. It's used to train deep neural networks using an architecture called convolutional layers. I'm not going to get into the details of convolutional networks here, but if you want to learn more about them, please [watch this](https://www.youtube.com/watch?v=2-Ol7ZB0MmU).\n",
    "\n",
    "Once trained, these models work astonishingly well as feature detectors for images they weren't trained on. Using a pre-trained network on images not in the training set is called transfer learning. Here we'll use transfer learning to train a network that can classify our cat and dog photos with near perfect accuracy.\n",
    "\n",
    "With `torchvision.models` you can download these pre-trained networks and use them in your applications. We'll include `models` in our imports now."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "%config InlineBackend.figure_format = 'retina'\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "import torch\n",
    "from torch import nn\n",
    "from torch import optim\n",
    "import torch.nn.functional as F\n",
    "from torchvision import datasets, transforms, models"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Most of the pretrained models require the input to be 224x224 images. Also, we'll need to match the normalization used when the models were trained. Each color channel was normalized separately, the means are `[0.485, 0.456, 0.406]` and the standard deviations are `[0.229, 0.224, 0.225]`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dir = 'Cat_Dog_data'\n",
    "\n",
    "# TODO: Define transforms for the training data and testing data\n",
    "train_transforms = transforms.Compose([transforms.RandomRotation(30),\n",
    "                                       transforms.RandomResizedCrop(224),\n",
    "                                       transforms.RandomHorizontalFlip(),\n",
    "                                       transforms.ToTensor(),\n",
    "                                       transforms.Normalize([0.5, 0.5, 0.5], \n",
    "                                                            [0.5, 0.5, 0.5])])\n",
    "\n",
    "test_transforms = transforms.Compose([transforms.RandomRotation(30),\n",
    "                                       transforms.RandomResizedCrop(224),\n",
    "                                       transforms.RandomHorizontalFlip(),\n",
    "                                       transforms.ToTensor(),\n",
    "                                       transforms.Normalize([0.5, 0.5, 0.5], \n",
    "                                                            [0.5, 0.5, 0.5])])\n",
    "\n",
    "# Pass transforms in here, then run the next cell to see how the transforms look\n",
    "train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)\n",
    "test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)\n",
    "\n",
    "trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)\n",
    "testloader = torch.utils.data.DataLoader(test_data, batch_size=64)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can load in a model such as [DenseNet](http://pytorch.org/docs/0.3.0/torchvision/models.html#id5). Let's print out the model architecture so we can see what's going on."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "model = models.densenet121(pretrained=True)\n",
    "model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This model is built out of two main parts, the features and the classifier. The features part is a stack of convolutional layers and overall works as a feature detector that can be fed into a classifier. The classifier part is a single fully-connected layer `(classifier): Linear(in_features=1024, out_features=1000)`. This layer was trained on the ImageNet dataset, so it won't work for our specific problem. That means we need to replace the classifier, but the features will work perfectly on their own. In general, I think about pre-trained networks as amazingly good feature detectors that can be used as the input for simple feed-forward classifiers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Freeze parameters so we don't backprop through them\n",
    "for param in model.parameters():\n",
    "    param.requires_grad = False\n",
    "\n",
    "from collections import OrderedDict\n",
    "classifier = nn.Sequential(OrderedDict([\n",
    "                          ('fc1', nn.Linear(1024, 500)),\n",
    "                          ('relu', nn.ReLU()),\n",
    "                          ('fc2', nn.Linear(500, 2)),\n",
    "                          ('output', nn.LogSoftmax(dim=1))\n",
    "                          ]))\n",
    "    \n",
    "model.classifier = classifier"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "With our model built, we need to train the classifier. However, now we're using a **really deep** neural network. If you try to train this on a CPU like normal, it will take a long, long time. Instead, we're going to use the GPU to do the calculations. The linear algebra computations are done in parallel on the GPU leading to 100x increased training speeds. It's also possible to train on multiple GPUs, further decreasing training time.\n",
    "\n",
    "PyTorch, along with pretty much every other deep learning framework, uses [CUDA](https://developer.nvidia.com/cuda-zone) to efficiently compute the forward and backwards passes on the GPU. In PyTorch, you move your model parameters and other tensors to the GPU memory using `model.to('cuda')`. You can move them back from the GPU with `model.to('cpu')` which you'll commonly do when you need to operate on the network output outside of PyTorch. As a demonstration of the increased speed, I'll compare how long it takes to perform a forward and backward pass with and without a GPU."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for device in ['cpu', 'cuda']:\n",
    "\n",
    "    criterion = nn.NLLLoss()\n",
    "    # Only train the classifier parameters, feature parameters are frozen\n",
    "    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)\n",
    "\n",
    "    model.to(device)\n",
    "\n",
    "    for ii, (inputs, labels) in enumerate(trainloader):\n",
    "\n",
    "        # Move input and label tensors to the GPU\n",
    "        inputs, labels = inputs.to(device), labels.to(device)\n",
    "\n",
    "        start = time.time()\n",
    "\n",
    "        outputs = model.forward(inputs)\n",
    "        loss = criterion(outputs, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        if ii==3:\n",
    "            break\n",
    "        \n",
    "    print(f\"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You can write device agnostic code which will automatically use CUDA if it's enabled like so:\n",
    "```python\n",
    "# at beginning of the script\n",
    "device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "...\n",
    "\n",
    "# then whenever you get a new Tensor or Module\n",
    "# this won't copy if they are already on the desired device\n",
    "input = data.to(device)\n",
    "model = MyModule(...).to(device)\n",
    "```\n",
    "\n",
    "From here, I'll let you finish training the model. The process is the same as before except now your model is much more powerful. You should get better than 95% accuracy easily.\n",
    "\n",
    ">**Exercise:** Train a pretrained models to classify the cat and dog images. Continue with the DenseNet model, or try ResNet, it's also a good model to try out first. Make sure you are only training the classifier and the parameters for the features part are frozen."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## TODO: Use a pretrained model to classify the cat and dog images\n",
    "# Use GPU if it's available\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "model = models.densenet121(pretrained=True)\n",
    "\n",
    "# Freeze parameters so we don't backprop through them\n",
    "for param in model.parameters():\n",
    "    param.requires_grad = False\n",
    "    \n",
    "model.classifier = nn.Sequential(nn.Linear(1024, 256),\n",
    "                                 nn.ReLU(),\n",
    "                                 nn.Dropout(0.2),\n",
    "                                 nn.Linear(256, 2),\n",
    "                                 nn.LogSoftmax(dim=1))\n",
    "\n",
    "criterion = nn.NLLLoss()\n",
    "\n",
    "# Only train the classifier parameters, feature parameters are frozen\n",
    "optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)\n",
    "\n",
    "model.to(device);\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "epochs = 1\n",
    "steps = 0\n",
    "running_loss = 0\n",
    "print_every = 5\n",
    "for epoch in range(epochs):\n",
    "    for inputs, labels in trainloader:\n",
    "        steps += 1\n",
    "        # Move input and label tensors to the default device\n",
    "        inputs, labels = inputs.to(device), labels.to(device)\n",
    "        \n",
    "        optimizer.zero_grad()\n",
    "        \n",
    "        logps = model.forward(inputs)\n",
    "        loss = criterion(logps, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        running_loss += loss.item()\n",
    "        \n",
    "        if steps % print_every == 0:\n",
    "            test_loss = 0\n",
    "            accuracy = 0\n",
    "            model.eval()\n",
    "            with torch.no_grad():\n",
    "                for inputs, labels in testloader:\n",
    "                    inputs, labels = inputs.to(device), labels.to(device)\n",
    "                    logps = model.forward(inputs)\n",
    "                    batch_loss = criterion(logps, labels)\n",
    "                    \n",
    "                    test_loss += batch_loss.item()\n",
    "                    \n",
    "                    # Calculate accuracy\n",
    "                    ps = torch.exp(logps)\n",
    "                    top_p, top_class = ps.topk(1, dim=1)\n",
    "                    equals = top_class == labels.view(*top_class.shape)\n",
    "                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()\n",
    "                    \n",
    "            print(f\"Epoch {epoch+1}/{epochs}.. \"\n",
    "                  f\"Train loss: {running_loss/print_every:.3f}.. \"\n",
    "                  f\"Test loss: {test_loss/len(testloader):.3f}.. \"\n",
    "                  f\"Test accuracy: {accuracy/len(testloader):.3f}\")\n",
    "            running_loss = 0\n",
    "            model.train()"
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
