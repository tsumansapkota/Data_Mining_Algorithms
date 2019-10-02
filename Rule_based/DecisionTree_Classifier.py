import numpy as np
import pandas as pd
import random

import seaborn as sns
import matplotlib.pyplot as plt

random.seed(0)


def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = int(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


# #### Data Pure ?

def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_threshold = 1

    for column in df.columns:
        unique_values = df[column].unique()
        example_value = unique_values[0]

        if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_threshold):
            feature_types.append(False)
        else:
            feature_types.append(True)

    return feature_types


def class_counts(data):
    labels = data[:, -1]
    counts = {}  # a dictionary of label -> count.
    for label in labels:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def classify_data(data):
    counts = class_counts(data)
    maxval = 0
    high_c = None
    for clas in counts:
        if counts[clas] > maxval:
            high_c = clas
            maxval = counts[clas]

    classification = high_c
    return classification


def get_potential_splits(data, continuous):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):

        values = data[:, column_index]
        unique_values = np.unique(values)

        continu = continuous[column_index]
        if continu:
            potential_splits[column_index] = []
            for index in range(1, len(unique_values)):
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]

                potential_split = (current_value + previous_value) / 2

                potential_splits[column_index].append(potential_split)
        else:
            potential_splits[column_index] = unique_values
    return potential_splits


def split_data(data, split_column, split_value, continuous):
    split_column_values = data[:, split_column]

    conti = continuous[split_column]
    if conti:
        data_below = data[split_column_values <= split_value]
        #     data_above = data[~(split_column_values <= split_value)]
        data_above = data[split_column_values > split_value]
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]

    return data_below, data_above


def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilitis = counts / counts.sum()
    entropy = np.sum(probabilitis * -np.log2(probabilitis))
    return entropy


def calculate_overall_entropy(data_below, data_above):
    n_data_points = len(data_below) + len(data_above)

    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points

    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))

    return overall_entropy


def gini(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    impurity = 1
    probabilitis = counts / counts.sum()
    for p in probabilitis:
        impurity -= p ** 2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    lef = p * gini(left)
    righ = p * (1 - p) * gini(right)

    return current_uncertainty - lef - righ


def determine_best_split(data, potential_splits, continuous):
    overall_entropy = 999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value,
                                                continuous=continuous)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


class Question:

    def __init__(self, column, value, header=None, continuous=False):
        self.column = column
        self.value = value
        self.continuous = continuous
        self.header = header

    def match(self, row):
        val = row[self.column]
        if self.continuous:
            return val <= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if self.continuous:
            condition = "<="

        if self.header is None:
            header = self.column
        else:
            header = self.header
        return "Is %s %s %s?" % (
            header, condition, str(self.value))


class Leaf:
    def __init__(self, classes):
        maxval = 0
        high_c = None
        for clas in classes:
            if classes[clas] > maxval:
                high_c = clas
                maxval = classes[clas]
        self.predictions = high_c

    #         self.predictions = classes  # this is compared hence makes difference in the tree.

    def __repr__(self):
        return self.predictions


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def decision_tree_algorithm(df, continuous=[], column_headers=None, min_samples=2, max_depth=5):
    # data preparations
    if column_headers is None:
        column_headers = df.columns
        continuous = determine_type_of_feature(df)
        data = df.values
    else:
        data = df

    # BASE_CASE

    class_algo = class_counts
    #     class_algo = classify_data

    if check_purity(data) or len(data) < min_samples or max_depth == 0:
        classification = class_algo(data)
        return Leaf(classification)

    # helper functions
    potential_splits = get_potential_splits(data, continuous)
    split_column, split_value = determine_best_split(data, potential_splits, continuous)
    data_below, data_above = split_data(data, split_column, split_value, continuous)
    if len(data_above) == 0:
        classification = class_algo(data_below)
        return Leaf(classification)
    elif len(data_below) == 0:
        classification = class_algo(data_above)
        return Leaf(classification)


    # RECURSIVE_PART
    else:
        max_depth -= 1

        # instantiate sub-tree
        feature_name = column_headers[split_column]
        conti = continuous[split_column]
        question = Question(split_column, split_value, feature_name, continuous=conti)

        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, continuous, column_headers, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, continuous, column_headers, min_samples, max_depth)

        if isinstance(yes_answer, Leaf) and isinstance(no_answer, Leaf):
            if yes_answer.predictions == no_answer.predictions:
                return yes_answer

        return Decision_Node(question, yes_answer, no_answer)


def print_tree(node, spacing="    "):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing[:int(len(spacing) / 2)] + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + spacing)

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + spacing)


def classify_example(example, tree):
    if isinstance(tree, Leaf):
        return tree.predictions

    if tree.question.match(example):
        return classify_example(example, tree.true_branch)
    else:
        return classify_example(example, tree.false_branch)


def calculate_accuracy(df, tree):
    df['classification'] = df.apply(classify_example, axis=1, args=(tree,))
    df['classification_correct'] = df.classification == df.label

    accuracy = df.classification_correct.mean()

    return accuracy

if __name__=='main':
    df = pd.read_csv('./iris.csv')

    cols = list(df.columns)
    cols[-1] = 'label'
    df.columns = cols
    print(df.head())

    # training_data = [
    #     ['Green', 3, 'Apple'],
    #     ['Yellow', 3, 'Apple'],
    #     ['Red', 1, 'Grape'],
    #     ['Red', 1, 'Grape'],
    #     ['Yellow', 3, 'Lemon'],
    # ]
    # tdf = pd.DataFrame.from_records(training_data, columns=['Color', 'Size', 'label'])


    train_df, test_df = train_test_split(df, test_size=0.2)
    tree = decision_tree_algorithm(train_df, max_depth=3)
    accuracy = calculate_accuracy(test_df, tree)

    print_tree(tree)

    print(accuracy)
