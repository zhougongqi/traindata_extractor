import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import tree
import os
import time
import logging

my_logger = logging.getLogger(__name__)


def DecisionTree(
    training_dataSet: np.array,
    model_folder: str,
    *,
    crossValidation_num: int = 3,
    split_ratio: float = 0.2,
    max_depth: int = 10,
    criterion_type: str = "entropy",
    label_str: str = "",
) -> str:
    """
    Function:
        Training decision tree model
    Input:
        training_data: np.array, feature followed by label (shape = [recordNum, featureNum + 1])
        model_folder: string, folder path to save the model file
        crossValidation_num: int, optional (default = 3), number of cross validation
        split_ratio: float, optional (default = 0.2), ratio of record number in cross validation
        max_depth: int, optional (default = 10), max depth of decision tree
        criterion_type: string, optional (default = "entropy")
                        “gini” for the Gini impurity and “entropy” for the information gain
        label_str: string, optional (default = ""), result model name label
    Output:
        model_path: string, full path of decision tree model file
                    if is None, training model failed
    """

    # check if all parameters are valid
    assert 0 < split_ratio < 1, "split_ratio must be in (0, 1)"
    assert 5 <= max_depth <= 100, "max_depth must be in [5, 100]"
    assert criterion_type in [
        "gini",
        "entropy",
    ], "criterion_type must be 'gini' or 'entropy'"
    assert os.path.exists(model_folder), "model_folder must be existed"
    assert 0 <= crossValidation_num <= 10, "crossValidation_num must be in [0, 10]"

    my_logger.info("Decision tree model training...")

    # split train feature and label
    train_feature = training_dataSet[:, :-1]
    train_label = training_dataSet[:, -1]

    # cross validation
    my_logger.info("Cross validation...")
    for loop_index in range(crossValidation_num):
        print("No.", loop_index + 1)
        train_sub_feature, test_sub_feature, train_sub_label, test_sub_label = train_test_split(
            train_feature, train_label, test_size=split_ratio
        )
        DT_model = tree.DecisionTreeClassifier(
            max_depth=max_depth, criterion=criterion_type
        )
        DT_model = DT_model.fit(train_sub_feature, train_sub_label)

        print("Features weight:", (DT_model.feature_importances_))

        print(
            "Training accuracy: %f"
            % (DT_model.score(train_sub_feature, train_sub_label))
        )

        print(
            "Testing accuracy: %f" % (DT_model.score(test_sub_feature, test_sub_label))
        )
        print(
            "Confusion matrix:\n",
            confusion_matrix(test_sub_label, DT_model.predict(test_sub_feature)),
        )

    # get final model
    my_logger.info("Final model training...")
    DT_model = tree.DecisionTreeClassifier(
        max_depth=max_depth, criterion=criterion_type
    )
    DT_model = DT_model.fit(train_feature, train_label)
    print("Training accuracy:%f" % (DT_model.score(train_feature, train_label)))

    # save model
    model_name = (
        "DecisionTree_"
        + label_str
        + time.strftime("%Y%m%d%H%M%S", time.localtime())
        + ".m"
    )
    model_path = os.path.join(model_folder, model_name)

    try:
        joblib.dump(DT_model, model_path)
        my_logger.success("Decision tree model saved! Result path:: %s", model_path)

    except Exception as e:
        my_logger.error("Save decision tree model failed!")
        return None

    return model_path


if __name__ == "__main__":

    label_1 = np.ones((1000, 1))
    features_1 = np.random.randn(1000, 5)
    traindata_1 = np.concatenate((features_1, label_1), axis=1)

    label_2 = 2 * np.ones((1000, 1))
    features_2 = np.random.randn(1000, 5)
    traindata_2 = np.concatenate((features_2, label_2), axis=1)

    trainDataSet = np.concatenate((traindata_1, traindata_2), axis=0)

    model_res_path = DecisionTree(trainDataSet, "/home/xyz/data_pool/U-TMP")
