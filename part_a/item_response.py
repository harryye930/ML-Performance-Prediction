from utils import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    theta_beta_array =[]
    for k, q in enumerate(data["question_id"]):
        i = data["user_id"][k]
        j = q
        theta_beta_array.append(theta[i]-beta[j])
    sigmoid_theta_beta_array = sigmoid(theta_beta_array)
    likelihood_array = np.multiply(np.array(sigmoid_theta_beta_array), np.array(data["is_correct"])) + np.multiply((1-np.array(sigmoid_theta_beta_array)), (1-np.array(data["is_correct"])))
    log_lklihood=np.sum(np.log(likelihood_array))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    beta_grad = np.zeros(beta.shape)
    theta_grad = np.zeros(theta.shape)
    theta_beta_array =[]
    for k, q in enumerate(data["question_id"]):
        i = data["user_id"][k]
        j = q
        theta_beta_array.append(theta[i]-beta[j])
    der_log_likelihood = -1.0*sigmoid(np.array(theta_beta_array)) + np.array(data["is_correct"])
    for k, q in enumerate(data["question_id"]):
        i = data["user_id"][k]
        j = q
        theta_grad[i]-=der_log_likelihood[k]
        beta_grad[j]+=der_log_likelihood[k]
    theta = theta - lr*theta_grad
    beta = beta - lr*beta_grad
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(max(max(data["user_id"]),len(set(data["user_id"])))+1)
    beta = np.zeros(max(max(data["question_id"]), len(set(data["question_id"])))+1)

    val_acc_lst = []
    lld_train = []
    lld_valid = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        lld_train.append(-1*neg_lld)
        lld_valid.append(-1*neg_log_likelihood(val_data, theta=theta, beta=beta))
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, lld_train, lld_valid


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    iterations = 15
    lr = 0.01
    theta, beta, val_acc_lst, lld_train, lld_valid = irt(train_data, val_data, lr, iterations)
    score = evaluate(data=test_data, theta=theta, beta=beta)
    print("Final Testing acc: {}\nFinal Validation acc:{}".format(score, val_acc_lst[-1])) 
    
    x = list(range(iterations))
    plt.plot(x, lld_train, label="Train Log-Likelihood")
    
    plt.xlabel('# of Iterations')
    # Set the y axis label of the current axis.
    plt.ylabel('Log Likelihood')
    
    # Set a title of the current axes.
    plt.title('Log-Likelihood for training over 15 iterations')
    
    # show a legend on the plot
    plt.legend()
    
    # Display a figure.
    plt.show()
    
    plt.plot(x, lld_valid, label="Validation Log-Likelihood")
    
    plt.xlabel('# of Iterations')
    # Set the y axis label of the current axis.
    plt.ylabel('Log Likelihood')
    # Set a title of the current axes.
    plt.title('Log-Likelihood for Validation over 15 iterations')
    # show a legend on the plot
    plt.legend()
    # Display a figure.

    plt.show()
    
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    
    a=np.argmax(beta)
    b=np.argmin(beta)
    c=1771
    beta_a=beta[a]
    beta_b=beta[b]
    beta_c=beta[c]
    x = []
    plt1=[]
    plt2=[]
    plt3=[]
    min=-7.5
    max=7.5
    iter = 10000
    for i in range(iter):
        theta = min + i*(max-min)/iter
        x.append(theta)
        plt1.append(theta-beta_a)
        plt2.append(theta-beta_b)
        plt3.append(theta-beta_c)
    
    plt.plot(x, sigmoid(np.array(plt1)), label="Question {}".format(a))
    plt.plot(x, sigmoid(np.array(plt2)), label="Question {}".format(b))
    plt.plot(x, sigmoid(np.array(plt3)), label="Question {}".format(c))
    plt.xlabel('Theta')
    # Set the y axis label of the current axis.
    plt.ylabel('Probability of correct answer')
    # Set a title of the current axes.
    plt.title('Probability of correct answer as a function of theta for question {}, {} and {}'.format(a, b, c))
    # show a legend on the plot
    plt.legend()
    # Display a figure.

    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
