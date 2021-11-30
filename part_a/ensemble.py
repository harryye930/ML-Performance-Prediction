import os.path

import numpy as np
from sklearn.impute import KNNImputer
from torch.autograd import Variable
import torch
import torch.optim as optim
import pandas as pd
from utils import *
from neural_network import AutoEncoder
import item_response as irt
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

def bagging_dict(train_dict):
    """
    Select without replacement
    """

    N = len(train_dict['user_id'])
    max_user = max(train_dict['user_id'])
    max_question = max(train_dict['question_id'])
    num_std_loss = int(N * 1/3.0)
    random_indeies = np.random.choice(range(0, N), size=num_std_loss, replace=True)
    ret_train_data = {'user_id':[max_user], 'question_id':[max_question], 'is_correct':[0]}

    for row in random_indeies:
        ret_train_data['user_id'].append(train_dict['user_id'][row])
        ret_train_data['question_id'].append(train_dict['question_id'][row])
        ret_train_data['is_correct'].append(train_dict['is_correct'][row])

    return ret_train_data
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
    result_train, valid_acc = evaluate_nn(model, zero_train_matrix, valid_data)
    result_test, test_acc = evaluate_nn(model, zero_train_matrix, test_data)

    print(f"valid acc: {valid_acc}, test acc: {test_acc}")
    return result_train, result_test


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

def predict_irt(data, theta, beta):
    """
    return predictions, given theta, beta and data
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = irt.sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.array(pred)

if __name__ == "__main__":
    train_matrix, valid_data, test_data = load_data()

    train_matrix_bagged = bagging(train_matrix)
    result_knn_valid = eval_knn_base_models(11, train_matrix_bagged, valid_data)
    result_knn_test = eval_knn_base_models(11, train_matrix_bagged, test_data)
    print(f"KNN accuracy: {evaluate_ensemble(valid_data, result_knn_valid)}")
    print(f"KNN accuracy: {evaluate_ensemble(test_data, result_knn_test)}")

    train_matrix_bagged = bagging(train_matrix)
    result_knn_valid = eval_knn_base_models(11, train_matrix_bagged, valid_data)
    result_knn_test = eval_knn_base_models(11, train_matrix_bagged, test_data)
    print(f"KNN accuracy: {evaluate_ensemble(valid_data, result_knn_valid)}")
    print(f"KNN accuracy: {evaluate_ensemble(test_data, result_knn_test)}")

    train_matrix_bagged = bagging(train_matrix)
    result_knn_valid = eval_knn_base_models(11, train_matrix_bagged, valid_data)
    result_knn_test = eval_knn_base_models(11, train_matrix_bagged, test_data)
    print(f"KNN accuracy: {evaluate_ensemble(valid_data, result_knn_valid)}")
    print(f"KNN accuracy: {evaluate_ensemble(test_data, result_knn_test)}")

    train_matrix_bagged = bagging(train_matrix)
    result_nn1_valid, result_nn1_test = eval_neural_net_base_model(train_matrix_bagged, valid_data, test_data, epoch=18, k=10)
    print(f"Neural Net 1 accuracy: {evaluate_ensemble(valid_data, result_nn1_valid)}")
    print(f"Neural Net 1 accuracy: {evaluate_ensemble(test_data, result_nn1_test)}")

    train_matrix_bagged = bagging(train_matrix)
    result_nn2_valid, result_nn2_test = eval_neural_net_base_model(train_matrix_bagged, valid_data, test_data, epoch=18, k=10)
    print(f"Neural Net 2 accuracy: {evaluate_ensemble(valid_data, result_nn2_valid)}")
    print(f"Neural Net 2 accuracy: {evaluate_ensemble(test_data, result_nn2_test)}")

    train_matrix_bagged = bagging(train_matrix)
    result_nn3_valid, result_nn3_test = eval_neural_net_base_model(train_matrix_bagged, valid_data, test_data, epoch=18, k=10)
    print(f"Neural Net 3 accuracy: {evaluate_ensemble(valid_data, result_nn3_valid)}")
    print(f"Neural Net 3 accuracy: {evaluate_ensemble(test_data, result_nn3_test)}")

    train_data_dict = load_train_csv("../data")
    bagged_train_dict = bagging_dict(train_data_dict)
    theta, beta, val_acc_lst = irt.irt(bagged_train_dict, valid_data, 0.03, 20)
    valid_pred = predict_irt(valid_data, theta, beta)
    test_pred = predict_irt(test_data, theta, beta)
    print(f"IRT valid accuracy: {evaluate_ensemble(valid_data, valid_pred)}")
    print(f"IRT test accuracy: {evaluate_ensemble(test_data, test_pred)}")
    
    ensemble_predictions = np.asmatrix([result_nn1_valid, valid_pred, result_knn_valid])
    average_predictions = np.asarray(ensemble_predictions.mean(axis=0))[0]
    validation_accuracy = evaluate_ensemble(valid_data, average_predictions)
    print(validation_accuracy)

    ensemble_predictions = np.asmatrix([result_nn1_test, test_pred, result_knn_test])
    average_predictions = np.asarray(ensemble_predictions.mean(axis=0))[0]
    test_accuracy = evaluate_ensemble(test_data, average_predictions)
    print(test_accuracy)


