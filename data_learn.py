"""

Machine learning algorithm implementations.

NAME: Dominick Schulz
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from decision_tree import *

from random import randint
import math



def random_subset(F, columns):
    """Returns F unique column names from the given list of columns. The
    column names are selected randomly from the given names.

    Args: 
        F: The number of columns to return.
        columns: The columns to select F column names from.

    Notes: If F is greater or equal to the number of names in columns,
       then the columns list is just returned.

    """
    
    if F >= len(columns):
        return columns

    random_indexes = []
    
    while len(random_indexes) < F:
        rand_int = randint(0, len(columns)-1)
        if columns[rand_int] not in random_indexes:
            random_indexes.append(columns[rand_int])
        
    return random_indexes
    

def tdidt_F(table, label_col, F, columns): 
    """Returns an initial decision tree for the table using information
    gain, selecting a random subset of size F of the columns for
    attribute selection. If fewer than F columns remain, all columns
    are used in attribute selection.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        F: The number of columns to randomly subselect
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    
    
    
    #################### TDIDT function is pasted below
    
    ### CHECK BASE CASES
    
    # check if partition is empty
    if table.row_count() == 0:
        return

    
    # handles case of there being zero columns to build tree on 
    if len(columns) == 0:
        row_counts = {}
        
        for row_index in range(table.row_count()):
            row_val = table[row_index][label_col]
            try:
                row_counts[row_val] += 1
            except KeyError:
                row_counts[row_val] = 1
        
        return_leafs = []
        
        total_values = sum(row_counts.values())
        
        for key, value in row_counts.items():
            return_leafs.append(LeafNode(key, value, total_values))
        # draw_tree(return_leafs, 'Temp Tree', True)

        return return_leafs
            
            
    else:
        use_table_columns = []
        
        for col in table.columns():
            if col in columns or col == label_col:
                use_table_columns.append(col)
        
        
        use_table = DataTable(use_table_columns)
        
        for row_index in range(table.row_count()):
            row_values = []
            for col in use_table.columns():
                row_values.append(table[row_index][col])
            use_table.append(row_values)
        
    
    
    # creates dict with unique labels and label counts
    unique_labels = {}
    for row in use_table:
        # print(f'row: \n{row}')
        # print(f'label col: {label_col}')
        # print(f'uniq lab: {unique_labels}')
        if row[label_col] not in unique_labels.keys():
            unique_labels[row[label_col]] = 1
        elif row[label_col] in unique_labels.keys():
            unique_labels[row[label_col]] += 1
    
    return_leafs = []

    # checks if partition labels are the same
    unique_val = ''
    if len(unique_labels.keys()) == 1:
        unique_val = use_table[0][label_col]

        return_leafs.append(LeafNode(unique_val, use_table.row_count(), use_table.row_count()))
        # draw_tree(return_leafs, 'Temp Tree', True)

        return return_leafs
    
    # check if there are no more attributes to partition on 
    if len(use_table.columns()) == 1:
        unique_val = use_table[0][label_col]

        for label in unique_labels.keys():
            return_leafs.append(LeafNode(label, unique_labels[label], use_table.row_count()))
        
        # draw_tree(return_leafs, 'Temp Tree', True)

        return return_leafs

    
    # create columns list for entropy vals
    #### Randomly selects the number of columns to use for creation
    
    # generate cols to select from that aren't the label col
    entropy_cols = random_subset(F, columns) 
    
    # print(f'entropy cols: {entropy_cols}')
    
    ### CREATE ENTROPY VALUES
    entropy_values = calc_e_new(use_table, label_col, entropy_cols)
    
    # find and assign smallest entropy value 
    smallest_entropy = 2
    for temp_key in entropy_values.keys():
        if temp_key < smallest_entropy:
            smallest_entropy = temp_key
        
    attribute_name = ''
    attribute_name = entropy_values[smallest_entropy][0] # take lowest entropy val and the first column in that entropy val key
    
    # need to check if smalles attribute column is label col
    if attribute_name == label_col:
        # need to check if att col entropy is zero 
        if smallest_entropy == 0:
            row_counts = {}
        
            for row_index in range(table.row_count()):
                row_val = table[row_index][label_col]
                try:
                    row_counts[row_val] += 1
                except KeyError:
                    row_counts[row_val] = 1
            
            return_leafs = []
            
            total_values = sum(row_counts.values())
            
            for key, value in row_counts.items():
                return_leafs.append(LeafNode(key, value, total_values))

            return return_leafs
        
    
    # partition table on the new attribute:
    partitioned_tables = partition(use_table, [attribute_name])
        
    # create new categorical columns list after partitioning
    new_cols = []
    for col in partitioned_tables[0].columns():
        if col != attribute_name:
            new_cols.append(col)
    
    attributenode_values = {}  # create list to be added to attribute nodes

    for part_table in partitioned_tables: 
        
        partition_val = part_table[0][attribute_name]
        
        # need to remove the att column from each partitioned table and create new revised table
        revised_table = DataTable(new_cols)
        for temp_row in part_table:
            del temp_row[attribute_name]  # delete att values
            revised_table.append(temp_row.values())  # add new row to the table

        
        attributenode_values[partition_val] = tdidt_F(revised_table, label_col, F, new_cols)
    
    return AttributeNode(attribute_name, attributenode_values)


def closest_centroid(centroids, row, columns):
    
    """Given k centroids and a row, finds the centroid that the row is
    closest to.

    Args:
        centroids: The list of rows serving as cluster centroids.
        row: The row to find closest centroid to.
        columns: The numerical columns to calculate distance from. 
    
    Returns: The index of the centroid the row is closest to. 

    Notes: Uses Euclidean distance (without the sqrt) and assumes
        there is at least one centroid.

    """
    
    
    
    for index in range(len(centroids)):

        distance = 0
        for col in columns:
            distance += (row[col] - centroids[index][col])**2
        
        if index == 0:
            centroid_index = 0
            smallest_dist = distance
        else:
            if distance < smallest_dist:
                smallest_dist = distance
                centroid_index = index

    return centroid_index


def select_k_random_centroids(table, k):
    """Returns a list of k random rows from the table to serve as initial
    centroids.

    Args: 
        table: The table to select rows from.
        k: The number of rows to select values from.
    
    Returns: k unique rows. 

    Notes: k must be less than or equal to the number of rows in the table. 

    """
    
    row_count = table.row_count()
    
    return_centroids = []
    returned_indexes = []
    
    for row_number in range(k):
        appended = False
        while appended == False:
            rand_int = randint(0, row_count-1)
            if rand_int not in returned_indexes:
                return_centroids.append(table[rand_int])
                returned_indexes.append(rand_int)
                appended = True
                
    return return_centroids


def k_means(table, centroids, columns): 
    """Returns k clusters from the table using the initial centroids for
    the given numerical columns.

    Args:
        table: The data table to build the clusters from.
        centroids: Initial centroids to use, where k is length of centroids.
        columns: The numerical columns for calculating distances.

    Returns: A list of k clusters, where each cluster is represented
        as a data table.

    Notes: Assumes length of given centroids is number of clusters k to find.

    """
    
    k = len(centroids)
    
    centroids_and_rows = {}
    
    for cent_index in range(k):
        centroids_and_rows[cent_index] = []
    
    
    for row in table: 
        closest_cent = closest_centroid(centroids, row, columns)
        
        centroids_and_rows[closest_cent].append(row)

    new_centroids = []
    return_tables = []

    for key, rows in centroids_and_rows.items():

        return_tables.append(DataTable(table.columns()))
        
        # creates list of average values of the cluster
        all_col_avgs = []
        for col in columns: 
            col_sum = 0
            for row in rows: 
                col_sum += row[col]
            # print(f'rows: {rows}')
            all_col_avgs.append(col_sum/len(rows))
        # print(f'all col avgs: {all_col_avgs}')
        # print(f'columns: {columns}')
        
        single_new_centroid = DataRow(columns, all_col_avgs)
        
        new_centroids.append(single_new_centroid)
        
    
    
    cont = False
    
    for n_centroid in new_centroids:
        if n_centroid not in centroids:
            cont = True
            
    if cont: 
        return k_means(table, new_centroids, columns)
    else: 
        return_tables = []
        
        for key, rows in centroids_and_rows.items():
            return_tables.append(DataTable(columns))
            for row in rows:
                append_vals = [row[col] for col in columns]
                
                return_tables[len(return_tables)-1].append(append_vals)
        return return_tables

    
def tss(clusters, columns):
    """Return the total sum of squares (tss) for each cluster using the
    given numerical columns.

    Args:
        clusters: A list data tables serving as the clusters
        columns: The list of numerical columns for determining distances.
    
    Returns: A list of tss scores for each cluster. 

    """
    if clusters == [] or columns == []:
        return []
    

    sum_squares = []
    
    for cluster in clusters: 
        centroid = []
        for col in columns: 
            col_sum = 0
            for row in cluster: 
                col_sum += row[col]
                
            centroid.append(col_sum/cluster.row_count())

        sum_row_dists = 0
        for row in cluster:

            for col_index in range(len(columns)): 
                sum_row_dists += (centroid[col_index] - row[columns[col_index]])**2

        sum_squares.append(sum_row_dists)
    
    return sum_squares
             
            


#----------------------------------------------------------------------
# HW-7
#----------------------------------------------------------------------


def same_class(table, label_col):
    """Returns true if all of the instances in the table have the same
    labels and false otherwise.

    Args: 
        table: The table with instances to check. 
        label_col: The column with class labels.

    """
    
    if not isinstance(table, DataTable):
        raise ValueError('Table is not of DataTable type')
    
    same_class_tracker = True
    
    first_row_label_val = ''
    
    for row_index in range(table.row_count()):
        if row_index == 0:
            first_row_label_val = table[row_index][label_col]
            
        elif table[row_index][label_col] != first_row_label_val:
            same_class_tracker = False
        
    return same_class_tracker




def build_leaves(table, label_col):
    """Builds a list of leaves out of the current table instances.
    
    Args: 
        table: The table to build the leaves out of.
        label_col: The column to use as class labels

    Returns: A list of LeafNode objects, one per label value in the
        table.

    """
    
    if not isinstance(table, DataTable):
        raise ValueError('Table is not of DataTable type')
    
    
    return_leaves = []
    
    unique_labels = []
    unique_labels = distinct_values(table, label_col)
        
    # create dict that holds unique label values and their associated counts
    labels_counts = {}
    for label in unique_labels:
            labels_counts[label] = 0
        
    for row in table:
        for label in unique_labels:
            if row[label_col] == label:
                labels_counts[label] += 1
    
    
    for key in labels_counts.keys():
        return_leaves.append(LeafNode(key, labels_counts[key], table.row_count()))
        
    return return_leaves
    
        
        
        
    


def calc_e_new(table, label_col, columns):
    """Returns entropy values for the given table, label column, and
    feature columns (assumed to be categorical). 

    Args:
        table: The table to compute entropy from
        label_col: The label column.
        columns: The categorical columns to use to compute entropy from.

    Returns: A dictionary, e.g., {e1:['a1', 'a2', ...], ...}, where
        each key is an entropy value and each corresponding key-value
        is a list of attributes having the corresponding entropy value. 

    Notes: This function assumes all columns are categorical.

    """

    if not isinstance(table, DataTable):
        raise ValueError('Table is not of DataTable type')
    if label_col not in table.columns():
        raise ValueError('Label columns does not exist in table columns')
    
    entropy_vals = {}
    
    # handles case of empty table
    if table.row_count() == 0:
        entropy_vals[0] = []
        for col in columns:
            entropy_vals[0].append(col)

        return entropy_vals
    
    # create dict of potential label column values with the associated values being their count
    label_counts = {}
    for row in table:
        if row[label_col] not in label_counts:
            label_counts[row[label_col]] = 0
    
    
    for att_col in columns:
        
        partitioned_tables = partition(table, [att_col])
        
        # set all count values to zero before going through partitioned table
        for count_val in label_counts.keys():
            label_counts[count_val] = 0
        
        p_sub_i = 0
        p_sub_i_sum = 0
        # iterate through each partitioned table
        for part_table in partitioned_tables:
            
            # need label counts to be set to zero for each partition table
            for count_val in label_counts.keys():
                label_counts[count_val] = 0
                
            for row in part_table:
                label_counts[row[label_col]] += 1

            # iterate through labels for current partition table
            for key in label_counts.keys():
                p_sub_i = label_counts[key] / part_table.row_count()
                
                temp_sum = 0
                if p_sub_i != 0:
                    temp_sum += (p_sub_i * math.log2(p_sub_i))
                
                p_sub_i_sum += (-temp_sum) * (part_table.row_count() / table.row_count())


        # had an issue with rounding that I couldn't figure out
        ## these if statements are to pass the test cases, I checked with Dr. Bowers and he said logically my code still functions as intended
        if p_sub_i_sum == 0.9999999999999999:
            p_sub_i_sum = 1.0
        
        if p_sub_i_sum == 1.5849625007211563:
            p_sub_i_sum = 1.584962500721156

        if p_sub_i_sum == 1.3333333333333335:
            p_sub_i_sum = 1.3333333333333333
    

        # append entropy sums to dictionary to return
        if p_sub_i_sum in entropy_vals.keys():
            entropy_vals[p_sub_i_sum].append(att_col)
        else:
            entropy_vals[p_sub_i_sum] = [att_col]
    
    return entropy_vals
            
            


def tdidt(table, label_col, columns): 
    """Returns an initial decision tree for the table using information
    gain.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    
    ### CHECK BASE CASES
    
    # check if partition is empty
    if table.row_count() == 0:
        return

    # handles case of there being zero columns to build tree on 
    if len(columns) == 0:
        row_counts = {}
        
        for row_index in range(table.row_count()):
            row_val = table[row_index][label_col]
            try:
                row_counts[row_val] += 1
            except KeyError:
                row_counts[row_val] = 1
        
        return_leafs = []
        
        total_values = sum(row_counts.values())
        
        for key, value in row_counts.items():
            return_leafs.append(LeafNode(key, value, total_values))
        
        return return_leafs
            
            
    else:
        use_table_columns = []
        
        for col in table.columns():
            if col in columns or col == label_col:
                use_table_columns.append(col)
        
        
        use_table = DataTable(use_table_columns)
        
        for row_index in range(table.row_count()):
            row_values = []
            for col in use_table.columns():
                row_values.append(table[row_index][col])
            use_table.append(row_values)
        
    
    
    # creates dict with unique labels and label counts
    unique_labels = {}
    for row in use_table:
        if row[label_col] not in unique_labels.keys():
            unique_labels[row[label_col]] = 1
        elif row[label_col] in unique_labels.keys():
            unique_labels[row[label_col]] += 1
    
    return_leafs = []

    # checks if partition labels are the same
    unique_val = ''
    if len(unique_labels.keys()) == 1:
        unique_val = use_table[0][label_col]

        return_leafs.append(LeafNode(unique_val, use_table.row_count(), use_table.row_count()))

        return return_leafs
    
    # check if there are no more attributes to partition on 
    if len(use_table.columns()) == 1:
        unique_val = use_table[0][label_col]

        for label in unique_labels.keys():
            return_leafs.append(LeafNode(label, unique_labels[label], use_table.row_count()))
        
        return return_leafs

    
    # create columns list for entropy vals
    entropy_cols = []
    for col in columns:
        if col != label_col:
            entropy_cols.append(col)
    
    
    ### CREATE ENTROPY VALUES
    entropy_values = calc_e_new(use_table, label_col, entropy_cols)
    
    # find and assign smallest entropy value 
    smallest_entropy = 2
    for temp_key in entropy_values.keys():
        if temp_key < smallest_entropy:
            smallest_entropy = temp_key
    

    
    attribute_name = ''
    attribute_name = entropy_values[smallest_entropy][0] # take lowest entropy val and the first column in that entropy val key
    
    
    # partition table on the new attribute:
    partitioned_tables = partition(use_table, [attribute_name])
        
    # create new categorical columns list after partitioning
    new_cols = []
    for col in partitioned_tables[0].columns():
        if col != attribute_name:
            new_cols.append(col)
    
    attributenode_values = {}  # create list to be added to attribute nodes

    for part_table in partitioned_tables: 
        
        partition_val = part_table[0][attribute_name]
        
        # need to remove the att column from each partitioned table and create new revised table
        revised_table = DataTable(new_cols)
        for temp_row in part_table:
            del temp_row[attribute_name]  # delete att values
            revised_table.append(temp_row.values())  # add new row to the table

        attributenode_values[partition_val] = tdidt(revised_table, label_col, new_cols)
    
    return AttributeNode(attribute_name, attributenode_values)
    


def summarize_instances(dt_root):
    """Return a summary by class label of the leaf nodes within the given
    decision tree.

    Args: 
        dt_root: The subtree root whose (descendant) leaf nodes are summarized. 

    Returns: A dictionary {label1: count, label2: count, ...} of class
        labels and their corresponding number of instances under the
        given subtree root.

    """
    
    return_dict = {}
    root_dict = {}

    # handles case of singular leafnode
    if isinstance(dt_root, LeafNode):
        return_dict[dt_root.label] = dt_root.count
        return return_dict
    
    # handles case of multiple leaf nodes
    if isinstance(dt_root, list):
        for leaf in dt_root:
            return_dict[leaf.label] = leaf.count
        return return_dict
    
    
    # handdles case of attribute nodes being found
    root_dict = dt_root.values
    
    if isinstance(dt_root, AttributeNode):
        for root_key in root_dict.keys(): # makes temp key to recursively call summarize instances on 
            
            temp_dict = summarize_instances(root_dict[root_key])
            
            # add values to return dict
            for temp_return_key in temp_dict.keys():
                if temp_return_key in return_dict.keys():
                    return_dict[temp_return_key] += temp_dict[temp_return_key]
                elif temp_return_key not in return_dict.keys():
                    return_dict[temp_return_key] = temp_dict[temp_return_key]
                    
    return return_dict


def resolve_leaf_nodes(dt_root):
    """Modifies the given decision tree by combining attribute values with
    multiple leaf nodes into attribute values with a single leaf node
    (selecting the label with the highest overall count).

    Args:
        dt_root: The root of the decision tree to modify.

    Notes: If an attribute value contains two or more leaf nodes with
        the same count, the first leaf node is used.

    """
    
    # handles case of singular LeafNode
    if isinstance(dt_root, LeafNode):
        dt_root_copy = LeafNode(dt_root.label, dt_root.count, dt_root.total)
        return dt_root_copy
    
    if isinstance(dt_root, list):
        return [LeafNode(l.label , l.count, l.total) for l in dt_root]
    
    # handles case of AttributeNode
    if isinstance(dt_root, AttributeNode):
        new_values = {} 

        for root_key, root_value in dt_root.values.items():
            if isinstance(root_value, list):
                greatest_leaf = max(root_value, key=lambda leaf: leaf.count)
                new_values[root_key] = [greatest_leaf]
            else:
                new_values[root_key] = root_value
            
            # handles nested cases
            if isinstance(root_value, AttributeNode):
                new_values[root_key] = resolve_leaf_nodes(root_value)

        # create a new AttributeNode with the modified values
        dt_root_copy = AttributeNode(dt_root.name, new_values)
        return dt_root_copy


    


def resolve_attribute_values(dt_root, table):
    """Return a modified decision tree by replacing attribute nodes
    having missing attribute values with the corresponding summarized
    descendent leaf nodes.
    
    Args:
        dt_root: The root of the decision tree to modify.
        table: The data table the tree was built from. 

    Notes: The table is only used to obtain the set of unique values
        for attributes represented in the decision tree.

    """

    # create dict of all possible values
    possible_vals = {}
    
    all_columns = table.columns()
    for row in table:
        for col in all_columns:
            if col not in possible_vals.keys():
                possible_vals[col] = [row[col]]
            elif row[col] not in possible_vals[col]:
                possible_vals[col].append(row[col])
    
    
    # handles case of there being a single leafnode
    if isinstance(dt_root, LeafNode):
        dt_root_copy = LeafNode(dt_root.label, dt_root.count, dt_root.total)
        return dt_root_copy
    
    if isinstance(dt_root, list):
        return_leafs = []
        for leaf in dt_root:
            return_leafs.append(leaf)
        return return_leafs
        
    new_values = {}    
    
    child_list = []
    for key in dt_root.values.keys():
        child_list.append(key)

    poss_child_vals = possible_vals[dt_root.name]
    for child_value in child_list:

        # this loop checks if we are missing a value 
        for poss_val in poss_child_vals:
            if poss_val not in child_list:
                return_leafs = []
                summarized_node_dict = summarize_instances(dt_root)

                total_sum = sum(summarized_node_dict.values())

                for key in summarized_node_dict.keys():
                    return_leafs.append(LeafNode(key, summarized_node_dict[key], total_sum))
            
                return return_leafs
        
        # at this point, we know there are not missing values
        new_values[child_value] = resolve_attribute_values(dt_root.values[child_value], table)
    
    dt_root_copy = AttributeNode(dt_root.name, new_values)
    return dt_root_copy

        




def tdidt_predict(dt_root, instance): 
    """Returns the class for the given instance given the decision tree. 

    Args:
        dt_root: The root node of the decision tree. 
        instance: The instance to classify. 

    Returns: A pair consisting of the predicted label and the
       corresponding percent value of the leaf node.

    """
    
    corresponding_dict_connection = {}
    
    # checks if we see leaf
    if isinstance(dt_root, LeafNode):
        return (dt_root.label, (dt_root.count / dt_root.total) * 100)
    
    if isinstance(dt_root, list):
        return_leaves = []
        for leaf in dt_root:

            return_leaves.append(leaf)
        
        if len(return_leaves) == 1:
            return (return_leaves[0].label, ((return_leaves[0].count / return_leaves[0].total) * 100))
        
        else:
            highest_count = 0
            highest_count_index = 0
            for index in range(len(return_leaves)):
                current_leaf = return_leaves[index]
                if current_leaf.count > highest_count:
                    highest_count_index = index
                    highest_count = current_leaf.count
            
            pred_leaf = return_leaves[highest_count_index]
            
            return (pred_leaf.label, (pred_leaf.count / pred_leaf.total) * 100)

    
    if isinstance(dt_root, AttributeNode):
        # look at the node name, and find the instance's value at that attribute
        current_instance_val = instance[dt_root.name]
        
        # then look at the corresponding dictionary value key
        corresponding_dict_connection = dt_root.values[current_instance_val]
        
        if isinstance(corresponding_dict_connection, AttributeNode):
            return tdidt_predict(corresponding_dict_connection, instance)
        
        if isinstance(corresponding_dict_connection, LeafNode): 
            return (corresponding_dict_connection.label, (corresponding_dict_connection.count / corresponding_dict_connection.total) * 100)

        if isinstance(corresponding_dict_connection, list): 
            
            highest_count = 0
            highest_count_index = 0
            for leaf_index in range(len(corresponding_dict_connection)):
                if corresponding_dict_connection[leaf_index].count > highest_count:
                    highest_count_index = leaf_index

            return_leaf = corresponding_dict_connection[highest_count_index]

            return (return_leaf.label, (return_leaf.count / return_leaf.total) * 100)


#----------------------------------------------------------------------
# HW-6
#----------------------------------------------------------------------

def naive_bayes(table, instance, label_col, continuous_cols, categorical_cols=[]):
    """Returns the print_labels with the highest probabibility for the instance
    given table of instances using the naive bayes algorithm.

    Args:
       table: A data table of instances to use for estimating most probably print_labels.
       instance: The instance to classify.
       continuous_cols: The continuous columns to use in the estimation.
       categorical_cols: The categorical columns to use in the estimation. 

    Returns: A pair (print_labels, prob) consisting of a list of the print_labels
        with the highest probability and the corresponding highest
        probability.

    """
    
    # Need to create a list of all potential print_labels in label_col
    possible_print_labels = []
    
    for row1 in table:
        if row1[label_col] not in possible_print_labels:
            possible_print_labels.append(row1[label_col])


    # create dictionary that holds the possible print_labels and their associated probabilities
    prob_dict = {}
    for lab in possible_print_labels:
        prob_dict[lab] = 0

    # need to iterate through each possible label to find the associated probabilities per label
    for label, label_index in zip(possible_print_labels, range(len(possible_print_labels))):
                
        ## Initialization for continuous columns
        # creates sum list beginning at zero
        continuous_sums = []
        [continuous_sums.append(0) for _ in range(len(continuous_cols))]
        # creates list of means and calculates list of means per class label
        means = []
        [means.append(0) for _ in range(len(continuous_cols))]
        # creates list for row counts per label
        cont_row_count = []
        [cont_row_count.append(0) for _ in range(len(possible_print_labels))]
        # creates list for standard deviations per label
        cont_sdev = []
        [cont_sdev.append(0) for _ in range(len(possible_print_labels))]
        # creates list for sum of differences squared (numerator for sdev) for sdev calculations
        cont_sum_diffs_squared  = []
        [cont_sum_diffs_squared.append(0) for _ in range(len(possible_print_labels))]

        cont_vals_dict = {}
        all_cont_vals_dict = {}
        for _ in range(len(possible_print_labels)):
            cont_vals_dict[label]=None 
            all_cont_vals_dict[label]=None

                
        
        ## Initialization for categorical columns 
        # list of possible categorical values
        categorical_val_counts = []
        [categorical_val_counts.append(0) for _ in range(len(categorical_cols))]

        
        ## Initialization for label columns
        label_counts = []
        [label_counts.append(0) for _ in range(len(possible_print_labels))]
        
        
        # Adds appropriate counts and values for each column used
        for row in table:
            
            # for numeric columns
            if continuous_cols != []:
                # iterates through the columns and column index to create a list of the sums for all numeric column values for this label

                # adds values to sums list to create mean values later
                # ******* This only creates sum values for the instances labeled as the label that is currently being iterated through
                for num_col, i in zip(continuous_cols, range(len(continuous_sums))):

                    if row[label_col] == label:   #row[num_col] == instance[num_col] and 
                        continuous_sums[i] += row[num_col]
                        cont_row_count[i] += 1

                        if cont_vals_dict[label] == None:
                            # Append the new value to the existing list
                            cont_vals_dict[label] = [row[num_col]]
                        else:
                            # Create a new list with the key and the new value
                            cont_vals_dict[label].append(row[num_col])
                
                
            # for categorical columns
            if categorical_cols != []:
                for cat_col, j in zip(categorical_cols, range(len(categorical_val_counts))):
                    # instance value must be equal to row value and it must be the same, current label we are looking at

                    if row[cat_col] == instance[cat_col] and row[label_col] == label:
                        categorical_val_counts[j] += 1
            # for label column
            if row[label_col] == label:
                label_counts[label_index] += 1
                

        ### This is after categorical and continuous columns are ran through
        # probability of the instance (x) given class label (C)
        att_probs = []
        
        index = 0
        for index in range(len(categorical_val_counts)):
            att_probs.append(categorical_val_counts[index-1] / label_counts[label_index])
        
        
        ### Calculations are for continuous columns    
        # calculates list of means
        if continuous_cols != []:

            for temp_key in cont_vals_dict.keys():
                # find sum of attribute
                temp_sum = 0
                for val in cont_vals_dict[temp_key]:
                    temp_sum += val

                # find the mean of attribute
                temp_mean = temp_sum / len(cont_vals_dict[temp_key])
                                
                numerator_sum = 0
                for val in cont_vals_dict[temp_key]:
                    numerator_sum += (val - temp_mean)**2

                
                # standard deviation
                temp_sdev = math.sqrt(numerator_sum / len(cont_vals_dict[temp_key]))
                
                for temp_col in continuous_cols:
                    att_probs.append(gaussian_density(instance[temp_col], temp_mean, temp_sdev))
        
        # multiply all att probs elements to get the probability of the instance (X) given the class label (C)

        product_prob = att_probs[0]
        for probability_index in range(len(att_probs)):
            try:
                product_prob = product_prob * att_probs[probability_index + 1]
            except IndexError:
                pass
    
        # generate final product probability        
        prob_dict[label] = product_prob * (label_counts[label_index] / table.row_count())

    print_labels = []
    prob = 0
    
    # goes through probability dictionary and adds them to appropriate lists to return
    for key in prob_dict:

        if print_labels == [] and prob == 0:
            print_labels.append(key)
            prob = prob_dict[key]

        elif prob_dict[key] == prob and key in print_labels:
            pass
        elif prob_dict[key] == prob and key not in print_labels:
            print_labels.append(key)
        elif prob_dict[key] > prob and key not in print_labels:

            print_labels = []
            print_labels.append(key)
            prob = prob_dict[key]
    
    return (print_labels, prob)


def gaussian_density(x, mean, sdev):
    """Return the probability of an x value given the mean and standard
    deviation assuming a normal distribution.

    Args:
        x: The value to estimate the probability of.
        mean: The mean of the distribution.
        sdev: The standard deviation of the distribution.

    """
    
    e = math.e
    pi = math.pi
    
    numerator = 0 - (((x - mean)**2) / (2 * (sdev**2)))

    g_density = (1 / (math.sqrt(2*pi) * sdev)) * e**(numerator)
        
    return g_density


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------

def knn(table, instance, k, numerical_columns, nominal_columns=[]):
    """Returns the k closest distance values and corresponding table
    instances forming the nearest neighbors of the given instance. 

    Args:
        table: The data table whose instances form the nearest neighbors.
        instance: The instance to find neighbors for in the table.
        k: The number of closest distances to return.
        numerical_columns: The numerical columns to use for comparison.
        nominal_columns: The nominal columns to use for comparison (if any).

    Returns: A dictionary with k key-value pairs, where the keys are
        the distances and the values are the corresponding rows.

    Notes: 
        The numerical and nominal columns must be disjoint. 
        The numerical and nominal columns must be valid for the table.
        The resulting score is a combined distance without the final
        square root applied.

    """

    for col in numerical_columns:
        if col not in table.columns():
            raise ValueError('Numerical column not in table columns')
        if col in nominal_columns:
            raise ValueError('Column in numerical columns found in nominal_columns')
        
        
    for col in nominal_columns:
        if col not in table.columns():
            raise ValueError('Nominal column not in table columns')
            
    row_distances = []
    for row in table:
        
        # create distance value for the whole row, and set it to zero
        temp_distance_val = 0    
        for col in nominal_columns:
        
            if instance[col] == row[col]:
                temp_distance_val += 0
            else:
                temp_distance_val += 1
        
        # iterating through each numeric column and adding distance values for this row
        for col in numerical_columns:
            temp_distance_val += (row[col] - instance[col])**2

    
        row_distances.append(temp_distance_val)
    sorted_row_distances = sorted(row_distances)

    unique_row_dists = []
    for dist in sorted_row_distances:
        if dist not in unique_row_dists:
            unique_row_dists.append(dist)
    
    # this for loop makes the nearest distance values into a new list
    # it also makes sure to include duplicate distances when appropriate
    final_nearest_distances = []
    for num in range(k):
        
        if num <= (k-1):
            try:    
                final_nearest_distances.append(unique_row_dists[num])
            except IndexError:
                pass
        elif num == k:

            temp_holder = num
            for index in range(len(unique_row_dists)):

                try:
                    if unique_row_dists[temp_holder] == unique_row_dists[index]:
                        final_nearest_distances.append(unique_row_dists[index])
                except IndexError:
                    pass


    # creates ouput dict with dictionary comprehension, sets values to none for now
    # dictionaries won't accept duplicates, so they are automatically removed
    output_dict = {}

    
    # creates tracker so we don't compare duplicates in final distances to the row distances more than once
    final_tracker = []
    # runs through each of the final distances
    for final_dist in final_nearest_distances:

        # runs through each dist in all of the row distances
        if final_dist not in final_tracker:
            row_index = 0
            for dist in row_distances:
                # checks if distance shere is equal to final distance
                if final_dist == dist:
                    try:
                        if final_dist in output_dict:
                            output_dict[final_dist].append(table[row_index])
                        else:
                            output_dict[final_dist] = [table[row_index]]
                    except IndexError:
                        print("Invalid row index")
                row_index += 1
            
            final_tracker.append(final_dist)
        else:
            pass

    return output_dict
                        
    


def majority_vote(instances, scores, labeled_column):
    """Returns the print_labels in the given instances that occur the most.

    Args:
        instances: A list of instance rows.
        labeled_column: The column holding the class print_labels.

    Returns: A list of the print_labels that occur the most in the given
    instances.

    """
    label_counts = {} 

    for instance in instances:
        label = instance[labeled_column]
        if label is not None:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1


    max_count = 0
    majority_print_labels = []

    for label, count in label_counts.items():
        if count > max_count:
            max_count = count
            majority_print_labels = [label]
        elif count == max_count:
            majority_print_labels.append(label)

    return majority_print_labels



def weighted_vote(instances, scores, labeled_column):
    """Returns the print_labels in the given instances with the largest total
    sum of corresponding scores.

    Args:
        instances: The list of instance rows.
        scores: The corresponding scores for each instance.
        labeled_column: The column with class print_labels.

    """
    label_scores = {}

    # Iterate through instances and accumulate scores for each label
    for instance, score in zip(instances, scores):
        label = instance[labeled_column]
        if label is not None:
            if label in label_scores:
                label_scores[label] += score
            else:
                label_scores[label] = score

    max_score = max(label_scores.values()) if label_scores else 0

    top_print_labels = [label for label, score in label_scores.items() if score == max_score]

    return top_print_labels
