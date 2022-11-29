import torch
import torch.nn as nn
import Data_Loaders as dl
import numpy as np


class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.input_to_hidden = nn.Linear(6, 300)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.hidden_to_hidden2 = nn.Linear(300, 300)
        self.hidden_to_hidden3 = nn.Linear(300, 300)
        self.hidden_to_hidden4 = nn.Linear(300, 30)
        self.hidden_to_hidden5 = nn.Linear(5, 5)
        self.hidden_to_hidden6 = nn.Linear(5, 5)
        self.hidden_to_hidden7 = nn.Linear(5, 5)
        self.hidden_to_hidden8 = nn.Linear(5, 5)
        self.hidden_to_output = nn.Linear(30, 1)
        self.dropout = nn.Dropout(0.3)
        pass

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden(input)
        hidden = self.relu(hidden)
        hidden = self.hidden_to_hidden2(hidden)
        hidden = self.relu(hidden)
        hidden = self.hidden_to_hidden3(hidden)
        hidden = self.relu(hidden)
        hidden = self.hidden_to_hidden4(hidden)
        hidden = self.relu(hidden)
        #hidden = self.dropout(hidden)
        output = self.hidden_to_output(hidden)
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        model_outputs = torch.empty((0,1))
        training_outputs = torch.empty((0,1))

        for idx, sample in enumerate(test_loader):
            model_outputs = torch.cat((model_outputs,model.forward(sample['input'])), 0)
            #print("training_outputs: ", training_outputs)
            #print("sample label: ", torch.unsqueeze(sample['label'],0))
            training_outputs = torch.cat((training_outputs,torch.unsqueeze(sample['label'],0)), 0)

        """ print("model_outputs: ", model_outputs)
        print("model output type: ", type(model_outputs))
        print("----------------------")
        print("training_outputs: ", training_outputs)
        print("training_outputs type: ", type(training_outputs)) """
        loss = loss_function(model_outputs, training_outputs)
        return loss

def main():
    model = Action_Conditioned_FF()
    """ action = 0
    batch_size = 16
    data_loaders = dl.Data_Loaders(batch_size)
    loss = model.evaluate(model, data_loaders.train_loader, nn.MSELoss())
    print("loss shape: ", loss.shape)
    print("loss: ", loss) """


if __name__ == '__main__':
    
    main()
    
