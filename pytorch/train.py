import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn

from model import LSTM

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(model_info['input_features'], model_info['hidden_dim'], model_info['num_layers'], model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# Gets training data in batches from the train.csv file
def _get_train_data(training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = train_data[[0]].to_numpy()
    train_x = train_data.drop([0], axis=1).to_numpy()
    
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)

    return (train_x, train_y)


# Provided training function
def train(model, train_data, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_data   - The data that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    x_train, y_train_lstm = train_data
    
    x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
    y_train_lstm = torch.from_numpy(y_train_lstm).type(torch.Tensor).to(device)
    
    lstm = []
    for t in range(epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    # Model Parameters
    
    parser.add_argument('--input_features', type=int, default=3, metavar='N', help='input batch size for training (default: 3)')
    parser.add_argument('--hidden_dim', type=int, default=64, metavar='N', help='size of the first hidden dimension (default: 64)')
    parser.add_argument('--num_layers', type=int, default=64, metavar='N', help='number of hidden layers')
    parser.add_argument('--output_dim', type=int, default=1, metavar='N', help='size of the output dimension (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=0.005, metavar='N', help='learning rate (default: 0.005)')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_data = _get_train_data(args.data_dir)

    model = LSTM(args.input_features, args.hidden_dim, args.num_layers, args.output_dim).to(device)

    ## Optimizer and loss function for training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss(reduction='mean')

    # Trains the model
    train(model, train_data, args.epochs, criterion, optimizer, device)

    # Model informations
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'output_dim': args.output_dim,
        }
        torch.save(model_info, f)
    

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
