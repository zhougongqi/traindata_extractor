import numpy as np
import os
import time
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, RegressorMixin


class skModel(BaseEstimator):
    def __init__(
        self,
        clf,
        training_dataSet: np.array,
        model_folder: str,
        *,
        label_str="test",
        model_path: str = None,
        C=0.8,
        kernel="rbf",
        gamma=20,
        decision_function_shape="ovr",
    ) -> str:
        """
        Sk-series classifier
        """
        self.clf = clf
        # initialize parameters
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.decision_function_shape = decision_function_shape

        # set traindata
        self.traindata = training_dataSet
        self.work_path = model_folder
        self.crossValidation_num = 5

        # split train feature and label
        self.train_feature = training_dataSet[:, :-1]
        self.train_label = training_dataSet[:, -1]
        print("feature shape: ", self.train_feature.shape)
        print("label shape: ", self.train_label.shape)

        # save model
        model_name = (
            self.clf.__class__.__name__
            + "_"
            + label_str
            + time.strftime("%Y%m%d%H%M%S", time.localtime())
            + ".m"
        )

        # check if a pretrained model is given
        if model_path is None:
            self.__has_model = False
            self.model_path = os.path.join(self.work_path, model_name)
        else:
            self.__has_model = True
            self.model_path = model_path
            self.clf = joblib.load(model_path)

    def fit(self):
        # check exist model
        if self.__has_model is True:
            print("exist model will be overwitten!")

        # # build svm classifier
        # clf = svm.SVC(
        #     # C=self.C,
        #     # kernel=self.kernel,
        #     # gamma=self.gamma,
        #     # decision_function_shape=self.decision_function_shape,
        # )
        self.test_fit()
        # fit the classifier~
        print("model {} fitting...".format(self.clf.__class__.__name__))
        self.clf.fit(self.train_feature, self.train_label)

        test_arr = np.array(
            [
                [140., 137., 269., 140., 3513., 867., 338.],
                [140., 137., 269., 140., 3513., 867., 338.],
            ]
        )
        print(self.clf.predict(test_arr))
        # self.clf = clf
        model_out_path = self.model_path
        joblib.dump(self.clf, model_out_path)
        return model_out_path

    def test_fit(self):
        for loop_index in range(self.crossValidation_num):
            print("No.", loop_index + 1)
            train_sub_feature, test_sub_feature, train_sub_label, test_sub_label = train_test_split(
                self.train_feature, self.train_label, test_size=0.7
            )
            # clf = svm.SVC(
            #     # C=self.C,
            #     # kernel=self.kernel,
            #     # gamma=self.gamma,
            #     # decision_function_shape=self.decision_function_shape,
            # )
            self.clf.fit(train_sub_feature, train_sub_label)

            # print("Features weight:", (DT_model.feature_importances_))

            print(
                "Training accuracy: %f"
                % (self.clf.score(train_sub_feature, train_sub_label))
            )

            print(
                "Testing accuracy: %f"
                % (self.clf.score(test_sub_feature, test_sub_label))
            )
            print(
                "Confusion matrix:\n",
                confusion_matrix(test_sub_label, self.clf.predict(test_sub_feature)),
            )
        return True
