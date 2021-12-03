
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch
import utils
import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.
    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = utils.load_train_sparse(base_path).toarray()
    valid_data = utils.load_valid_csv(base_path)
    test_data = utils.load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    #zero_train_matrix = fillin_na_as_mean(zero_train_matrix)
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    # GAUSSIAN NOISE
    zero_train_matrix = torch.FloatTensor(zero_train_matrix) #+ np.sqrt(0.1) * torch.randn(zero_train_matrix.shape)

    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k1, k2):
        """ Initialize a class AutoEncoder.
        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k1)
        self.h = nn.Linear(k1, k2)
        self.i = nn.Linear(k2, k1)
        self.j = nn.Linear(k1, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.
        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        i_w_norm = torch.norm(self.i.weight, 2) ** 2
        j_w_norm = torch.norm(self.j.weight, 2) ** 2


        return g_w_norm + h_w_norm #+ i_w_norm + j_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.
        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        g = nn.Sigmoid()
        h = nn.Sigmoid()
        i = nn.Sigmoid()
        j = nn.Sigmoid()

        out = j(self.j(i(self.i(h(self.h(g(self.g(inputs))))))))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_matrix, zero_train_data, train_data, valid_data, test_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.
    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # Tell PyTorch you are training the model.
    model.train()


    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_matrix.shape[0]
    train_acc = []
    valid_acc = []
    test_acc = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + (lamb/2)*model.get_weight_norm() # added regularizer
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        train_acc.append(evaluate(model, zero_train_data, train_data))
        valid_acc.append(evaluate(model, zero_train_data, valid_data))
        test_acc.append(evaluate(model, zero_train_data, test_data))
        print(epoch, train_acc[-1], valid_acc[-1], test_acc[-1])
    return train_acc, valid_acc, test_acc


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.
    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    # set seed to ensure results are reproducible
    np.random.seed(0)
    torch.manual_seed(0)
    train_data = utils.load_train_csv("../data")

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Set model hyperparameters.
    num_questions = zero_train_matrix.shape[1]

    model = AutoEncoder(num_questions, 100, 10)
    # Set optimization hyperparameters.
    lr = 0.05
    num_epoch = 50
    lamb = 0.001

    train_acc, valid_acc, test_acc = train(model, lr, lamb, train_matrix, zero_train_matrix, train_data, valid_data, test_data, num_epoch)

    x = list(range(num_epoch))
    plt.plot(x, train_acc, label="Train Acc")
    plt.plot(x, valid_acc, label="Valid Acc")
    plt.plot(x, test_acc, label="Test Acc")
    plt.xlabel('# of Epoch')
    # Set the y axis label of the current axis.
    plt.ylabel('Accuracy')
    # Set a title of the current axes.
    plt.title('Train, Validation, and Test Accuracy Over Epochs')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    #plt.savefig("NN_acc.png")
    plt.show()




    print(f"Test Acc: {evaluate(model, zero_train_matrix, test_data)}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()