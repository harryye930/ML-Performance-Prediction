import os.path

import numpy as np
from sklearn.impute import KNNImputer
from torch.autograd import Variable
import torch
import torch.optim as optim
import pandas as pd
from utils import *
from neural_network import AutoEncoder
import random


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
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)


    return train_matrix, valid_data, test_data


def bagging(train_matrix):
    """
    Select without replacement
    """

    N, K = train_matrix.shape
    num_std_loss = int(N * 1/3)
    random_indeies = np.random.choice(range(0, N), size=num_std_loss, replace=True)
    train_matrix = train_matrix.copy()

    for row in random_indeies:
        empty_array = np.full((1, K), np.NaN)
        train_matrix[row] = empty_array

    return train_matrix


def eval_knn_base_models(k, train_matrix_bagged, valid_data):
    """
    Implement KNN on bagged dataset evaluate the accuracy.
    """

    nbrs = KNNImputer(n_neighbors=k)  # best performing k = 11
    knn_result_matrix = nbrs.fit_transform(train_matrix_bagged)
    knn_results = sparse_matrix_predictions(valid_data, knn_result_matrix, threshold=0.5)
    return knn_results


def eval_neural_net_base_model(train_matrix_bagged, valid_data, test_data, epoch, k):
    """
    Setup hyperparameters for neural net, train and evaluate the accuracy
    """
    zero_train_matrix = train_matrix_bagged.copy()
    zero_train_matrix[np.isnan(train_matrix_bagged)] = 0
    train_matrix_bagged = torch.FloatTensor(train_matrix_bagged)
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)

    num_questions = train_matrix_bagged.shape[1]
    model = AutoEncoder(num_questions, k)
    lr = 0.05
    lamb = 0.001
    train_nn(model, lr, lamb, train_matrix_bagged, zero_train_matrix, epoch)
    result, valid_acc = evaluate_nn(model, zero_train_matrix, valid_data)
    result_test, test_acc = evaluate_nn(model, zero_train_matrix, test_data)

    print(f"valid acc: {valid_acc}, test acc: {test_acc}")
    return result


def train_nn(model, lr, lamb, train_data, zero_train_data, epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param epoch: int
    :return: None
    """
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            loss = torch.sum((output - target) ** 2.) + \
                   (lamb/2)*model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()


def evaluate_nn(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: Array of predictions and accuracy
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()
    result = []
    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        result.append(output[0][valid_data["question_id"][i]].item())
        total += 1

    return result, correct/total


def evaluate_ensemble(data, prediction):
    """
    Evaluate the prediction (List[int]) base on the valid/test data (dict)
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        if prediction[i] >= 0.5 and data["is_correct"][i]:
            total_accurate += 1
        if prediction[i] < 0.5 and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)
    # return evaluate(data, prediction)


if __name__ == "__main__":
    train_matrix, valid_data, test_data = load_data()

    train_matrix_bagged = bagging(train_matrix)
    result_knn = eval_knn_base_models(11, train_matrix_bagged, valid_data)
    print(f"KNN accuracy: {evaluate_ensemble(valid_data, result_knn)}")

    train_matrix_bagged = bagging(train_matrix)
    result_knn2 = eval_knn_base_models(16, train_matrix_bagged, valid_data)
    print(f"KNN2 accuracy: {evaluate_ensemble(valid_data, result_knn2)}")

    train_matrix_bagged = bagging(train_matrix)
    result_knn3 = eval_knn_base_models(6, train_matrix_bagged, valid_data)
    print(f"KNN3 accuracy: {evaluate_ensemble(valid_data, result_knn3)}")

    train_matrix_bagged = bagging(train_matrix)
    result_nn1 = eval_neural_net_base_model(train_matrix_bagged, valid_data, test_data, epoch=18, k=10)
    print(f"Neural Net 1 accuracy: {evaluate_ensemble(valid_data, result_nn1)}")

    train_matrix_bagged = bagging(train_matrix)
    result_nn2 = eval_neural_net_base_model(train_matrix_bagged, valid_data, test_data, epoch=18, k=10)
    print(f"Neural Net 2 accuracy: {evaluate_ensemble(valid_data, result_nn2)}")

    train_matrix_bagged = bagging(train_matrix)
    result_nn3 = eval_neural_net_base_model(train_matrix_bagged, valid_data, test_data, epoch=18, k=10)
    print(f"Neural Net 3 accuracy: {evaluate_ensemble(valid_data, result_nn3)}")

    ensemble_predictions = np.asmatrix([result_knn, result_knn2, result_knn3, result_nn1, result_nn2, result_nn3])
    average_predictions = np.asarray(ensemble_predictions.mean(axis=0))[0]

    validation_accuracy = evaluate_ensemble(valid_data, average_predictions)
    print(validation_accuracy)
    test_accuracy = evaluate_ensemble(test_data, average_predictions)
    print(test_accuracy)


