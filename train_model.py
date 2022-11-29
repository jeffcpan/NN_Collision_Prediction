from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF
import pickle

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()


    testlosses = []
    loss_function = nn.MSELoss()
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    testlosses.append(min_loss.item())

    losses = []

    #print(model.parameters())

    learning_rate = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iter = 0
    for epoch_i in range(no_epochs):
        model.train()
        optimizer.zero_grad()

        
        loss = model.evaluate(model, data_loaders.train_loader, loss_function)
        #print("loss type: ", type(loss))
        #print("loss: ", loss)
        #print("loss item: ", loss.item())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        print(f"epoch: {epoch_i}/{no_epochs}, trainloss: {loss}")

    model.eval()
    testloss = model.evaluate(model, data_loaders.test_loader, loss_function)
    print("testloss: ", testloss)

    #plt.plot(losses)
    #plt.show()

    torch.save(model.state_dict(), 'saved/saved_model.pkl', _use_new_zipfile_serialization=False)

    pass



if __name__ == '__main__':
    no_epochs = 400 #change this value
    train_model(no_epochs)
