import os
import numpy as np

"""splits data to three parts - train, test and validate
    
"""


def split(filename, train_size, test_size):
    all_data = _file_to_df(filename)
    (
        train,
        test,
        validate
    ) = (np.array([]), np.array([]), np.array([]))

    if all_data is not None :
        np.random.shuffle(all_data)
        train, test, validate = np.split(all_data,
                                         [int(train_size * len(all_data)),                  # train data
                                          int((train_size + test_size) * len(all_data))])   # test data

    print("Dataset '" + filename + "' has been successfully split into: "
        "train(" + str(len(train)) + "), "
        "test(" + str(len(test)) + ") and "
        "validate(" + str(len(validate)) + ")")

    return train, test, validate


def _file_to_df(filename):
    if not os.path.isfile(filename):
        print('file ' + filename + ' doesn\'t exist')
        return None
    else:
        file = open(filename)
        data = []
        """momentálně zpracovává data ve formátu r(a,b)
        """
        for line in file.read().splitlines():
            if not line == '':
                split_line = line.split("(")
                relation_name = split_line[0].strip()
                node_names = split_line[1].split(',')

                """remove extra characters"""
                node_names[0] = node_names[0].strip()
                node_names[1] = node_names[1].strip()
                node_names[1] = node_names[1].__str__().replace(')', '')

                data.append((node_names[0], relation_name, node_names[1]))

        return data