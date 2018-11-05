import numpy as np
import os
import time
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix


class SVMClassifier:
    def __init__(
        self,
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
        SVM classifier
        """
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
            "SVM_" + label_str + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".m"
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

        # build svm classifier
        clf = svm.SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            decision_function_shape=self.decision_function_shape,
        )
        self.test_fit()
        # fit the classifier~
        print("SVM fitting...")
        clf.fit(self.train_feature, self.train_label)
        self.clf = clf
        joblib.dump(clf, self.model_path)
        return True

    def test_fit(self):
        for loop_index in range(self.crossValidation_num):
            print("No.", loop_index + 1)
            train_sub_feature, test_sub_feature, train_sub_label, test_sub_label = train_test_split(
                self.train_feature, self.train_label, test_size=0.7
            )
            clf = svm.SVC(
                # C=self.C,
                kernel=self.kernel,
                # gamma=self.gamma,
                # decision_function_shape=self.decision_function_shape,
            )
            clf = clf.fit(train_sub_feature, train_sub_label)

            # print("Features weight:", (DT_model.feature_importances_))

            print(
                "Training accuracy: %f"
                % (clf.score(train_sub_feature, train_sub_label))
            )

            print(
                "Testing accuracy: %f" % (clf.score(test_sub_feature, test_sub_label))
            )
            print(
                "Confusion matrix:\n",
                confusion_matrix(test_sub_label, clf.predict(test_sub_feature)),
            )
        return True
