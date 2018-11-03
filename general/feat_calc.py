import numpy as np


def feat_calc(
    feat_array: np.array,
    index_1: int,
    index_2: int,
    calc_type: str,
    *,
    dst_index: int = -1,
) -> np.ndarray:
    """
    Function:
        Band math for two feature array, generate new feature array
        (Currently only supports binocular operations)
    Input:
        feat_array: np.array, original feature array, shape is
                    ((sample number), (feature nmber + label( = 1)))
        index_1: int, the first feature order involved in the calculation
        index_2: int, the second feature order involved in the calculation
        calc_type: str, calculation type, including "addition", "subtraction",
                    "multiplication", ""division""
        dst_index: int, optional (default = -1), location of the calculation
                    result in the new feature array
    Output:
        feat_array: np.array, new feature array with calculation result
    """

    assert calc_type in [
        "addition",
        "subtraction",
        "multiplication",
        "division",
    ], "calculation type error"

    assert feat_array.shape[1] >= 3, "feature number must be greater than 2"

    assert (
        index_1 < feat_array.shape[1] - 1 and index_2 < feat_array.shape[1] - 1
    ), "the index must be less than feature number"

    assert index_1 != index_2, "two indexes cannot be equal"

    assert (
        dst_index < feat_array.shape[1]
    ), "destination index must be less than feature number"

    # calculate new feature result
    calculation_res = {
        "addition": lambda a, b: a + b,
        "subtraction": lambda a, b: a - b,
        "multiplication": lambda a, b: a * b,
        "division": lambda a, b: a / b,
    }

    feat_res = calculation_res[calc_type](
        feat_array[:, index_1], feat_array[:, index_2]
    )

    # add new feature to ori feature array
    feat_array = np.insert(feat_array, dst_index, values=feat_res, axis=1)
    return feat_array


if __name__ == "__main__":
    ori_feature = np.array(
        [[1, 1, 1, 1], [2, 2, 2, 1], [3, 3, 3, 1], [4, 4, 4, 2], [5, 5, 5, 2]]
    )
    print(ori_feature.shape)
    index_1 = 0
    index_2 = 1
    calc_type = "division"
    dst_index = 3
    print(feat_calc(ori_feature, index_1, index_2, calc_type, dst_index=dst_index))
