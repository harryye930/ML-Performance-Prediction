from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt
import pandas as pd


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def compute_and_draw_val_accuracy(matrix, val_data, test_data):
    ks = [1, 6, 11, 16, 21, 26]
    result = []
    for k in ks:
        result.append(knn_impute_by_user(matrix, val_data, k))
    df_knn = pd.DataFrame({"k": ks,
                           "correctness": result})
    df_knn.plot(kind='bar', x='k')
    plt.savefig("knn_by_user.png")
    plt.show()

    max_k = ks[np.argmax(result)]
    test_acc = knn_impute_by_user(matrix, test_data, max_k)
    print(f"Testing accuracy on k={max_k}: {test_acc}")


    result = []
    for k in ks:
        result.append(knn_impute_by_item(matrix, val_data, k))
    df_knn = pd.DataFrame({"k": ks,
                           "correctness": result})
    df_knn.plot(kind='bar', x='k')
    plt.savefig("knn_by_user.png")
    plt.show()

    max_k = ks[np.argmax(result)]
    test_acc = knn_impute_by_item(matrix, test_data, max_k)
    print(f"Testing accuracy on k={max_k}: {test_acc}")




def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)




    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    compute_and_draw_val_accuracy(sparse_matrix, val_data, test_data)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
