"""Machine learning algorithm evaluation functions. 

NAME: Dominick Schulz
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from data_learn import *
from random import randint



#----------------------------------------------------------------------
# HW-8
#----------------------------------------------------------------------

def bootstrap(table): 
    """Creates a training and testing set using the bootstrap method.

    Args: 
        table: The table to create the train and test sets from.

    Returns: The pair (training_set, testing_set)

    """
    
    total_row_count = table.row_count()
    
    training_set = DataTable(table.columns())
    testing_set = DataTable(table.columns())
    
    # tuple of all training set values
    training_values = []
    
    while training_set.row_count() < total_row_count:
        rand_index = randint(0, total_row_count - 1)
        training_set.append(table[rand_index].values())
        training_values.append(table[rand_index].values())
        
    for row in table: 
        if row.values() not in training_values:
            testing_set.append(row.values())
    
    return (training_set, testing_set)
            

def stratified_holdout(table, label_col, test_set_size):
    """Partitions the table into a training and test set using the holdout
    method such that the test set has a similar distribution as the
    table.

    Args:
        table: The table to partition.
        label_col: The column with the class labels. 
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """

    test_set = DataTable(table.columns())
    training_set = DataTable(table.columns())    
    
    partitioned_tables = partition(table, [label_col])
    
    percentage_dict = {}
    
    for part_table in partitioned_tables:
        percentage = part_table.row_count() / table.row_count()
        
        percentage_dict[part_table[0][label_col]] = part_table.row_count()
        
        num_rows_to_add = math.ceil(percentage * test_set_size)

        partition_test_indexes = []
        
        for row_index in range(num_rows_to_add):
            appended = False

            while not appended:
                rand_int = randint(0, part_table.row_count() - 1)

                if rand_int not in partition_test_indexes:
                    test_set.append(part_table[rand_int].values())
                    partition_test_indexes.append(rand_int)
                    appended = True
            
        for row_index in range(part_table.row_count()):
            if row_index not in partition_test_indexes:
                training_set.append(part_table[row_index].values())

    return training_set, test_set


def tdidt_eval_with_tree(dt_root, test, label_col, label_vals):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       td_root: The decision tree to use.
       test: The testing data set.
       label_col: The column being predicted.
       label_vals: distinct, possible label column values

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    
    ### Creates and initializes confusion matrix
    # create list of all potential values in label_col
    label_col_vals = []
    
    for val in label_vals:
        label_col_vals.append(val)
        
    # create confusion matrix and initialize it with predicted columns
    matrix_columns = []
    matrix_columns.append('Actual')
    for col_index in range(len(label_vals)):
        matrix_columns.append(label_vals[col_index])
        
    confusion_matrix = DataTable(matrix_columns)    
    
    predicted_col_vals = label_vals
    for actual_val in label_vals:
        row_values = [actual_val]
        # print(f'inital row: {row_values}')
        for pred_val in predicted_col_vals:
            row_values.append(0)
        
        # appends the generated row to the matrix table
        append_row = DataRow(matrix_columns, row_values)
        confusion_matrix.append(append_row.values())
            
    for instance in test:
        (predicted_label, confidence) = tdidt_predict(dt_root, instance)
        actual_label = instance[label_col]

        for row_index in range(len(label_vals)):
            
            if label_vals[row_index] == actual_label:
                confusion_matrix.update(row_index, predicted_label, (confusion_matrix[row_index][predicted_label] + 1))
                    
    
    return confusion_matrix


def random_forest(table, remainder, F, M, N, label_col, columns):
    """Returns a random forest build from the given table. 
    
    Args:
        table: The original table for cleaning up the decision tree.
        remainder: The table to use to build the random forest.
        F: The subset of columns to use for each classifier.
        M: The number of unique accuracy values to return.
        N: The total number of decision trees to build initially.
        label_col: The column with the class labels.
        columns: The categorical columns used for building the forest.

    Returns: A list of (at most) M pairs (tree, accuracy) consisting
        of the "best" decision trees and their corresponding accuracy
        values. The exact number of trees (pairs) returned depends on
        the other parameters (F, M, N, and so on).

    """    
        
    bootstrap_training_validation = []
    
    for _ in range(N):
        bootstrap_training_validation.append(bootstrap(remainder))
        while not bootstrap_training_validation[0][0].row_count():
            bootstrap_training_validation.append(bootstrap(remainder))

    possible_labels = []
    for row in remainder: 
        if row[label_col] not in possible_labels:
            possible_labels.append(row[label_col])
    
    
    initial_tree_list = []
    accuracies_list = []
    
    
    # NEED TO USE TDIDT_F
    for train_valid in bootstrap_training_validation:
        current_tree = tdidt_F(train_valid[0], label_col, F, columns)
        
        current_tree = resolve_attribute_values(current_tree, table)
        current_tree = resolve_leaf_nodes(current_tree)
        
        confusion_matrix = tdidt_eval_with_tree(current_tree, train_valid[1], label_col, distinct_values(remainder, label_col))        
        
        temp_acc_list = []
        
        for row in confusion_matrix:
            temp_accuracy = accuracy(confusion_matrix, row['Actual'])
            temp_acc_list.append(temp_accuracy)
         
        # append average accuracy to overall accuracy list
        accuracies_list.append(sum(temp_acc_list) / len(temp_acc_list))

        # append tree to tree list 
        initial_tree_list.append(current_tree)
        

    if len(initial_tree_list) != len(accuracies_list):
        raise ValueError('length of initial_tree_list != length of accuracies_list')

    combined_acc_trees = list(zip(accuracies_list, initial_tree_list))

    sorted_combined = sorted(combined_acc_trees, key=lambda x: x[0], reverse=True)
    
    sorted_acc, sorted_trees = zip(*sorted_combined)
    
    final_list = []
    for index in range(len(sorted_acc)):
        if len(final_list) >= M:
            return final_list
        else:
            final_list.append((sorted_trees[index], sorted_acc[index]))
                
    return final_list


def random_forest_eval(table, train, test, F, M, N, label_col, columns):
    """Builds a random forest and evaluate's it given a training and
    testing set.

    Args: 
        table: The initial table.
        train: The training set from the initial table.
        test: The testing set from the initial table.
        F: Number of features (columns) to select.
        M: Number of trees to include in random forest.
        N: Number of trees to initially generate.
        label_col: The column with class labels. 
        columns: The categorical columns to use for classification.

    Returns: A confusion matrix containing the results. 

    Notes: Assumes weighted voting (based on each tree's accuracy) is
        used to select predicted label for each test row.

    """
    
    ## Initialize confusion matrix
    label_col_vals = []
    
    for row in table:
        if row[label_col] not in label_col_vals:
            label_col_vals.append(row[label_col])
        
    # create confusion matrix and initialize it with predicted columns
    matrix_columns = []
    matrix_columns.append('Actual')
    for col_index in range(len(label_col_vals)):
        matrix_columns.append(label_col_vals[col_index])
        
    confusion_matrix = DataTable(matrix_columns)
        
    predicted_col_vals = label_col_vals
    for actual_val in label_col_vals:
        row_values = [actual_val]
        for pred_val in predicted_col_vals:
            row_values.append(0)
        
        # appends the generated row to the matrix table
        append_row = DataRow(matrix_columns, row_values)
        confusion_matrix.append(append_row.values())
    
    
    # create trees and accuracies list
    trees_accuracies = random_forest(table, train, F, M, N, label_col, columns)
    # make predictions for every row in test and add it to the confusion matrix
    for row in test: 
        pred_label_acc = []
        for tree_acc in trees_accuracies:
            tree = tree_acc[0]
            acc = tree_acc[1]
            
            (pred_label, confidence) = tdidt_predict(tree, row)
            
            pred_label_acc.append([pred_label, acc])
            
        predictions_weights = {}
        for prediction_acc in pred_label_acc:
            weight = prediction_acc[1]
            prediction = prediction_acc[0]
            
            if prediction not in predictions_weights.keys():
                predictions_weights[prediction] = weight
            else: 
                predictions_weights[prediction] = predictions_weights[prediction] + weight
        
        sorted_predictions_weights = dict(sorted(predictions_weights.items(), key=lambda item: item[1], reverse=True))
                
        actual_label = row[label_col]
        predicted_label = next(iter(sorted_predictions_weights.keys()))
        
        for row_index in range(len(label_col_vals)):
            
            if label_col_vals[row_index] == actual_label:
                confusion_matrix.update(row_index, predicted_label, (confusion_matrix[row_index][predicted_label] + 1))
        
    return confusion_matrix
        



#----------------------------------------------------------------------
# HW-7
#----------------------------------------------------------------------


def tdidt_eval(train, test, label_col, columns):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # creates and cleans decision tree
    initial_tree = tdidt(train, label_col, columns)
    res_att_tree = resolve_attribute_values(initial_tree, train)
    final_tree = resolve_leaf_nodes(res_att_tree)

    return tdidt_eval_with_tree(final_tree, test, label_col, distinct_values(union_all([train, test]), label_col))    




def tdidt_stratified(table, k_folds, label_col, columns):
    """Evaluates tdidt prediction approach over the table using stratified
    k-fold cross validation, returning a single confusion matrix of
    the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        columns: The categorical columns for tdidt. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    
    ## Initialize confusion matrix
    label_col_vals = []
    
    for row in table:
        if row[label_col] not in label_col_vals:
            label_col_vals.append(row[label_col])
        
    # create confusion matrix and initialize it with predicted columns
    matrix_columns = []
    matrix_columns.append('Actual')
    for col_index in range(len(label_col_vals)):
        matrix_columns.append(label_col_vals[col_index])
        
    confusion_matrix = DataTable(matrix_columns)
        
    predicted_col_vals = label_col_vals
    for actual_val in label_col_vals:
        row_values = [actual_val]
        for pred_val in predicted_col_vals:
            row_values.append(0)
        
        # appends the generated row to the matrix table
        append_row = DataRow(matrix_columns, row_values)
        confusion_matrix.append(append_row.values())
    
    
    # Stratify the tables
    stratified_tables = DataTable(table.columns())
    stratified_tables = stratify(table, label_col, k_folds)
    
    
    for test_table_index in range(len(stratified_tables)):
        
        train_tables = []
        
        for train_table_index in range(len(stratified_tables)):
            # creates and cleans decision tree


            if train_table_index != test_table_index:
                train_tables.append(stratified_tables[train_table_index])
            
        train_set = union_all(train_tables)
        
        initial_tree = tdidt(train_set, label_col, columns)
        res_att_tree = resolve_attribute_values(initial_tree, train_set)
        final_tree = resolve_leaf_nodes(res_att_tree)
        
        
        
        # iterates through each row in test table
        for row in train_set:
            (prediction_val, confidence) = tdidt_predict(final_tree, row)
            actual_val = row[label_col]
            for row_index in range(len(label_col_vals)):

                if label_col_vals[row_index] == actual_val:
                    confusion_matrix.update(row_index, prediction_val, (confusion_matrix[row_index][prediction_val] + 1))
                
    return confusion_matrix


#----------------------------------------------------------------------
# HW-6
#----------------------------------------------------------------------

def stratify(table, label_column, k):
    """Returns a list of k stratified folds as data tables from the given
    data table based on the label column.

    Args:
        table: The data table to partition.
        label_column: The column to use for the label. 
        k: The number of folds to return. 

    Note: Does not randomly select instances for the folds, and
        instead produces folds in order of the instances given in the
        table.

    """ 
    
    unique_labels = []
    unique_labels = distinct_values(table, label_column)
    
    
    return_folds = []

    
    for table_num in range(k):
        return_folds.append(DataTable(table.columns()))
    

    for current_label in unique_labels:
        i = 1
        for row in table:
            if row[label_column] == current_label:
                return_folds[i-1].append(row.values())
                if i % k == 0:
                    i = 1
                else:
                    i += 1
                
    return return_folds
            




def union_all(tables):
    """Returns a table containing all instances in the given list of data
    tables.

    Args:
        tables: A list of data tables. 

    Notes: Returns a new data table that contains all the instances of
       the first table in tables, followed by all the instances in the
       second table in tables, and so on. The tables must have the
       exact same columns, and the list of tables must be non-empty.

    """
    
    if tables == []:
        raise ValueError('No tables were entered')
    
    # creates list to make sure all columns are the same 
    table_cols = []
    for col in tables[0].columns():
        table_cols.append(col)
    
    unioned_tables = DataTable(table_cols)
    
    for table in tables:
        if table.columns() != table_cols:
            raise ValueError('Invalid columns, columns are not equal')

        for row in table:
            unioned_tables.append(row.values())
    
    return unioned_tables


def naive_bayes_eval(train, test, label_col, continuous_cols, categorical_cols=[]):
    """Evaluates the given test set using naive bayes over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       continuous_cols: The continuous columns (estimated via PDF)
       categorical_cols: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    
    # create list of all potential values in label_col
    label_col_vals = []
    
    for row in train:
        if row[label_col] not in label_col_vals:
            label_col_vals.append(row[label_col])
    for row in test:
        if row[label_col] not in label_col_vals:
            label_col_vals.append(row[label_col])
        
    
    confusion_matrix = DataTable(matrix_columns)

    
    # create confusion matrix and initialize it with predicted columns
    matrix_columns = []
    matrix_columns.append('Actual')
    for col_index in range(len(label_col_vals)):
        matrix_columns.append(label_col_vals[col_index])
        
    predicted_col_vals = label_col_vals
    for actual_val in label_col_vals:
        row_values = [actual_val]
        for pred_val in predicted_col_vals:
            row_values.append(0)
        
        # appends the generated row to the matrix table
        append_row = DataRow(matrix_columns, row_values)
        confusion_matrix.append(append_row.values())
            
    
    for row in test:
        prediction_val = naive_bayes(train, row, label_col, continuous_cols, categorical_cols)[0][0]
        actual_val = row[label_col]

        for row_index in range(len(label_col_vals)):
            
            if label_col_vals[row_index] == actual_val:
                confusion_matrix.update(row_index, prediction_val, (confusion_matrix[row_index][prediction_val] + 1))
    
    return confusion_matrix
    
    
    

def naive_bayes_stratified(table, k_folds, label_col, cont_cols, cat_cols=[]):
    """Evaluates naive bayes over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        cont_cols: The continuous columns for naive bayes. 
        cat_cols: The categorical columns for naive bayes. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    
    ## Initialize confusion matrix
    label_col_vals = []
    
    for row in table:
        if row[label_col] not in label_col_vals:
            label_col_vals.append(row[label_col])
        
    # create confusion matrix and initialize it with predicted columns
    matrix_columns = []
    matrix_columns.append('Actual')
    for col_index in range(len(label_col_vals)):
        matrix_columns.append(label_col_vals[col_index])
        
    confusion_matrix = DataTable(matrix_columns)
        
    predicted_col_vals = label_col_vals
    for actual_val in label_col_vals:
        row_values = [actual_val]
        for pred_val in predicted_col_vals:
            row_values.append(0)
        
        # appends the generated row to the matrix table
        append_row = DataRow(matrix_columns, row_values)
        confusion_matrix.append(append_row.values())
    
    
    # Stratify the tables
    stratified_tables = DataTable(table.columns())
    stratified_tables = stratify(table, label_col, k_folds)    
    
    for test_table_index in range(len(stratified_tables)):
        
        train_tables = []
        
        for train_table_index in range(len(stratified_tables)):
            if train_table_index != test_table_index:
                train_tables.append(stratified_tables[train_table_index])
        train_set = union_all(train_tables)
        
        # iterates through each row in test table
        for row in stratified_tables[test_table_index]:
            prediction_val = naive_bayes(train_set, row, label_col, cont_cols, cat_cols)[0][0]
            actual_val = row[label_col]

            for row_index in range(len(label_col_vals)):
                
                if label_col_vals[row_index] == actual_val:
                    confusion_matrix.update(row_index, prediction_val, (confusion_matrix[row_index][prediction_val] + 1))
                
    return confusion_matrix
            
        


def knn_stratified(table, k_folds, label_col, vote_fun, k, num_cols, nom_cols=[]):
    """Evaluates knn over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        vote_fun: The voting function to use with knn.
        num_cols: The numeric columns for knn.
        nom_cols: The nominal columns for knn.

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
        
    ## Initialize confusion matrix
    label_col_vals = []
    
    for row in table:
        if row[label_col] not in label_col_vals:
            label_col_vals.append(row[label_col])
        
    # create confusion matrix and initialize it with predicted columns
    matrix_columns = []
    matrix_columns.append('Actual')
    for col_index in range(len(label_col_vals)):
        matrix_columns.append(label_col_vals[col_index])
        
    confusion_matrix = DataTable(matrix_columns)
        
    predicted_col_vals = label_col_vals
    for actual_val in label_col_vals:
        row_values = [actual_val]
        for pred_val in predicted_col_vals:
            row_values.append(0)
        
        # appends the generated row to the matrix table
        append_row = DataRow(matrix_columns, row_values)
        confusion_matrix.append(append_row.values())
    
    
    # Stratify the tables
    stratified_tables = DataTable(table.columns())
    stratified_tables = stratify(table, label_col, k_folds)
    
    for test_table_index in range(len(stratified_tables)):
        
        train_tables = []
        
        for train_table_index in range(len(stratified_tables)):
            if train_table_index != test_table_index:
                train_tables.append(stratified_tables[train_table_index])
        train_set = union_all(train_tables)
        
        
        # iterates through each row in test table
        for row in stratified_tables[test_table_index]:
            nearest_neighbors_dict = knn(train_set, row, k, num_cols, nom_cols)

            # extracts and makes a list of the rows in the dictionary, and scores list
            nearest_neighbors_rows = []
            scores = []
            for dist_val in nearest_neighbors_dict:
                scores.append(0 - dist_val)
                for row_index in range(len(nearest_neighbors_dict[dist_val])):
                    nearest_neighbors_rows.append(nearest_neighbors_dict[dist_val][row_index])
            
            
            # guesses label based on which function was inputted
            # prediction will just be the first label returned
            vote_label = []
            vote_label = vote_fun(nearest_neighbors_rows, scores, label_col)
            
            # compare guessed label to actual test value and add to specific 
            predicted_value = vote_label[0]
            actual_value = row[label_col]
            
            # print(f'pred val: {predicted_value}')
            # print(f'actual val: {actual_value}')
            
            for row_index in range(len(label_col_vals)):

                if label_col_vals[row_index] == actual_value:
                    confusion_matrix.update(row_index, predicted_value, (confusion_matrix[row_index][predicted_value] + 1))
                
    return confusion_matrix
    



#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------



def holdout(table, test_set_size):
    """Partitions the table into a training and test set using the holdout method. 

    Args:
        table: The table to partition.
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """

    test_set = DataTable(table.columns())
    training_set = DataTable(table.columns())    
    selected_indices = set()
    
    while test_set.row_count() < test_set_size:
        random_index = randint(0, table.row_count() - 1)
        if random_index not in selected_indices:
            selected_indices.add(random_index)
            test_set.append(table[random_index].values())
    
    for row_index, row in enumerate(table):
        if row_index not in selected_indices:
            training_set.append(row.values())
    
    return training_set, test_set
        


def knn_eval(train, test, vote_fun, k, label_col, numeric_cols, nominal_cols=[]):
    """Returns a confusion matrix resulting from running knn over the
    given test set. 

    Args:
        train: The training set.
        test: The test set.
        vote_fun: The function to use for selecting labels.
        k: The k nearest neighbors for knn.
        label_col: The column to use for predictions. 
        numeric_cols: The columns compared using Euclidean distance.
        nominal_cols: The nominal columns for knn (match or no match).

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the given voting function returns multiple labels, the
        first such label is selected.

    """
    
    ### CREATE AND INITIALIZE CONFUSION MATRIX WITH VALUES AT ZERO
    # create list of all potential values in label_col
    label_col_vals = []
    
    for row in train:
        if row[label_col] not in label_col_vals:
            label_col_vals.append(row[label_col])
    for row in test:
        if row[label_col] not in label_col_vals:
            label_col_vals.append(row[label_col])
        
        
    # create confusion matrix and initialize it with predicted columns
    matrix_columns = []
    matrix_columns.append('Actual')
    for col_index in range(len(label_col_vals)):
        matrix_columns.append(label_col_vals[col_index])
    
    confusion_matrix = DataTable(matrix_columns)

    # adds rows to the matrix table, but sets each value to zero
    predicted_col_vals = label_col_vals
    for actual_val in label_col_vals:
        row_values = [actual_val]
        for pred_val in predicted_col_vals:
            row_values.append(0)
        
        # appends the generated row to the matrix table
        append_row = DataRow(matrix_columns, row_values)
        confusion_matrix.append(append_row.values())
            
        
            
    
    ### ITERATE THROUGH TEST SET AND CREATE PREDICTIONS 
    for row in test:
        # saves output dictionary with knn key distances and rows
        nearest_neighbors_dict = knn(train, row, k, numeric_cols, nominal_cols)

        # extracts and makes a list of the rows in the dictionary, and scores list
        nearest_neighbors_rows = []
        scores = []
        for dist_val in nearest_neighbors_dict:
            scores.append(0 - dist_val)
            for row_index in range(len(nearest_neighbors_dict[dist_val])):
                nearest_neighbors_rows.append(nearest_neighbors_dict[dist_val][row_index])

        
        # guesses label based on which function was inputted
        # prediction will just be the first label returned
        vote_label = []
        vote_label = vote_fun(nearest_neighbors_rows, scores, label_col)
        
        # compare guessed label to actual test value and add to specific 
        predicted_value = vote_label[0]
        actual_value = row[label_col]


        actual_index = 0
        for row_index in range(len(label_col_vals)):
            
            if label_col_vals[row_index] == actual_value:
                confusion_matrix.update(row_index, predicted_value, (confusion_matrix[row_index][predicted_value] + 1))

    return confusion_matrix     





def accuracy(confusion_matrix, label):
    """Returns the accuracy for the given label from the confusion matrix.
    
    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the accuracy of.

    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    i = 0
    for col in confusion_matrix.columns():
        if i > 0:
            for row in confusion_matrix:    
                # checks if column is the label we are checking
                if col == label:
                    if row[confusion_matrix.columns()[0]] == col:
                        true_positives += row[col]
                    else:
                        false_positives += row[col]
                else:
                    if row[confusion_matrix.columns()[0]] == label:
                        false_negatives += row[col]
                    else:
                        true_negatives += row[col]
        i += 1   
    
    # print(f'(TP{true_positives} + TN{true_negatives}) / (TP{true_positives} + TN{true_negatives} + FP{false_positives} + FN{false_negatives})')
    try: 
        accuracy_num = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        
        return accuracy_num     
    except ZeroDivisionError:
        return 0 
        


def precision(confusion_matrix, label):
    """Returns the precision for the given label from the confusion
    matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the precision of.

    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    i = 0
    for col in confusion_matrix.columns():
        if i > 0:
            for row in confusion_matrix:    
                # checks if column is the label we are checking
                if col == label:
                    if row[confusion_matrix.columns()[0]] == col:
                        true_positives += row[col]
                    else:
                        false_positives += row[col]
                else:
                    if row[confusion_matrix.columns()[0]] == label:
                        false_negatives += row[col]
                    else:
                        true_negatives += row[col]
        i += 1   
    return_precision = true_positives / (true_positives + false_positives)
    return return_precision



def recall(confusion_matrix, label): 
    """Returns the recall for the given label from the confusion matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the recall of.

    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    i = 0
    for col in confusion_matrix.columns():
        if i > 0:
            for row in confusion_matrix:    
                # checks if column is the label we are checking
                if col == label:
                    if row[confusion_matrix.columns()[0]] == col:
                        true_positives += row[col]
                    else:
                        false_positives += row[col]
                else:
                    if row[confusion_matrix.columns()[0]] == label:
                        false_negatives += row[col]
                    else:
                        true_negatives += row[col]
        i += 1   
        
    return true_positives / (true_positives + false_negatives)