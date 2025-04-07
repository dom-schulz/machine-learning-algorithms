"""Data utility functions.

NAME: Dominick Schulz
DATE: Fall 2023
CLASS: CPSC 322

"""

from math import sqrt

from data_table import DataTable, DataRow
import matplotlib.pyplot as plt



def calculate_averages(result_confusion_matrix):
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
        
        try: 
            accuracy_num = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
            
            return accuracy_num     
        except ZeroDivisionError:
            return 0 
    
    def f_measure(matrix, label):
        precision_val = precision(matrix, label)
        recall_val = recall(matrix, label)
        return (2 * recall_val * precision_val) / (recall_val + precision_val)
    
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
    
    
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f_measure = 0
    num_classes = result_confusion_matrix.row_count()

    for row in result_confusion_matrix:
        try:
            accuracy_value = accuracy(result_confusion_matrix, row['Actual'])
            total_accuracy += accuracy_value
            
        except ZeroDivisionError:
            pass
            
        try:
            precision_value = precision(result_confusion_matrix, row['Actual'])
            total_precision += precision_value
            
        except ZeroDivisionError:
            pass
            
        try:
            recall_value = recall(result_confusion_matrix, row['Actual'])
            total_recall += recall_value

        except ZeroDivisionError:
            pass
        try:
            f_measure_value = f_measure(result_confusion_matrix, row['Actual'])
            total_f_measure += f_measure_value

        except ZeroDivisionError:
            pass

    # Calculate averages
    avg_accuracy = total_accuracy / num_classes
    avg_precision = total_precision / num_classes
    avg_recall = total_recall / num_classes
    avg_f_measure = total_f_measure / num_classes

    # Print averages
    print(f'Average Accuracy: {avg_accuracy}')
    print(f'Average Precision: {avg_precision}')
    print(f'Average Recall: {avg_recall}')
    print(f'Average F Measure: {avg_f_measure}')

    return



def normalize(table, column):
    """Normalize the values in the given column of the table. This
    function modifies the table.

    Args:
        table: The table to normalize.
        column: The column in the table to normalize.

    """
    
    column_values = [row[column] for row in table]

    minimum = min(column_values)
    maximum = max(column_values)
    col_range = maximum - minimum
    
    for row_index in range(table.row_count()):
        normalized_value = (table.rows([row_index])[0][column] - minimum) / col_range
        table.update(row_index, column, normalized_value)



def discretize(table, column, cut_points):
    """Discretize column values according to the given list of n-1
    cut_points to form n ordinal values from 1 to n. This function
    modifies the table.

    Args:
        table: The table to discretize.
        column: The column in the table to discretize.

    """
    
    ordered_values = [i + 1 for i in range(len(cut_points) + 1)]
    
    for row_index in range(table.row_count()):
        i = len(cut_points)

        while i > 0 and table[row_index][column] < cut_points[i - 1]:
            i -= 1

        table.update(row_index, column, ordered_values[i])
    
    


#----------------------------------------------------------------------
# HW4
#----------------------------------------------------------------------


def column_values(table, column):
    """Returns a list of the values (in order) in the given column.

    Args:
        table: The data table that values are drawn from
        column: The column whose values are returned
    
    """
    
    if column not in table.columns():
        raise IndexError('Invalid column name')
    
    return_list = []

    for row in table:
        return_list.append(row[column])

    return return_list



def mean(table, column):
    """Returns the arithmetic mean of the values in the given table
    column.

    Args:
        table: The data table that values are drawn from
        column: The column to compute the mean from

    Notes: 
        Assumes there are no missing values in the column.

    """
    
    if column not in table.columns():
        raise IndexError('Invalid column name')
    
    values_list = column_values(table, column)
    
    total_vals = sum(values_list)
    
    return total_vals / len(values_list)


def variance(table, column):
    """Returns the variance of the values in the given table column.

    Args:
        table: The data table that values are drawn from
        column: The column to compute the variance from

    Notes:
        Assumes there are no missing values in the column.

    """
    
    if column not in table.columns():
        raise IndexError('Invalid column name')
    
    values_list = column_values(table, column)
    
    average_val = mean(table, column)
    
    temp_sum = 0
    
    for val in values_list:
        if val == '':
            raise ValueError('Missing value detected, error') 
        temp_sum = temp_sum + ((val - average_val) ** 2)

    try:
        return temp_sum / len(values_list)
    except ZeroDivisionError:
        return 0
        
    
    


def std_dev(table, column):
    """Returns the standard deviation of the values in the given table
    column.

    Args:
        table: The data table that values are drawn from
        column: The colume to compute the standard deviation from

    Notes:
        Assumes there are no missing values in the column.

    """
    
    if column not in table.columns():
        raise IndexError('Invalid column name')
    
    return sqrt(variance(table, column))




def covariance(table, x_column, y_column):
    """Returns the covariance of the values in the given table columns.
    
    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x-values"
        y_column: The column with the "y-values"

    Notes:
        Assumes there are no missing values in the columns.        

    """
    
    if x_column not in table.columns() or y_column not in table.columns():
        raise IndexError('Invalid column name')
    
    x_vals = column_values(table, x_column)
    y_vals = column_values(table, y_column)
    
    x_mean = mean(table, x_column)
    y_mean = mean(table, y_column)
    
    for val in x_vals:
        if val == '':
            raise ValueError('Missing value detected, error')
    for val in y_vals:
        if val == '':
            raise ValueError('Missing value detected, error')
    
    temp_sum = 0

    for i in range(len(x_vals)):
        temp_xval = x_vals[i]
        temp_yval = y_vals[i]
        temp_sum += (temp_xval - x_mean) * (temp_yval - y_mean)
        
    return temp_sum / len(x_vals)


def linear_regression(table, x_column, y_column):
    """Returns a pair (slope, intercept) resulting from the ordinary least
    squares linear regression of the values in the given table columns.

    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"

    """
    if x_column not in table.columns() or y_column not in table.columns():
        raise IndexError('Invalid column name')
    
    x_vals = column_values(table, x_column)
    y_vals = column_values(table, y_column)
    
    x_mean = mean(table, x_column)
    y_mean = mean(table, y_column)
    
    num_temp_sum = 0

    for i in range(len(x_vals)):
        temp_xval = x_vals[i]
        temp_yval = y_vals[i]
        num_temp_sum += (temp_xval - x_mean) * (temp_yval - y_mean)
    
    den_temp_sum = 0
    
    for temp_xval in x_vals:
        if temp_xval == '':
            raise ValueError('Missing value detected, error') 
        den_temp_sum = den_temp_sum + ((temp_xval - x_mean) ** 2)

    m = num_temp_sum / den_temp_sum
    
    b = y_mean - (m * x_mean)
    
    return m, b

def correlation_coefficient(table, x_column, y_column):
    """Return the correlation coefficient of the table's given x and y
    columns.

    Args:
        table: The data table that value are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"

    Notes:
        Assumes there are no missing values in the columns.        

    """
    if x_column not in table.columns() or y_column not in table.columns():
        raise IndexError('Invalid column name')

    x_vals = column_values(table, x_column)
    y_vals = column_values(table, y_column)
    
    for val in x_vals:
        if val == '':
            raise ValueError('Missing value detected, error')
    for val in y_vals:
        if val == '':
            raise ValueError('Missing value detected, error')
    
    coefficient = covariance(table, x_column, y_column) / (std_dev(table, x_column) * std_dev(table, y_column))
    
    return coefficient


def frequency_of_range(table, column, start, end):
    """Return the number of instances of column values such that each
    instance counted has a column value greater or equal to start and
    less than end. 
    
    Args:
        table: The data table used to get column values from
        column: The column to bin
        start: The starting value of the range
        end: The ending value of the range

    Notes:
        start must be less than end

    """
    
    if column not in table.columns():
        raise IndexError('Invalid column name')
    
    if start > end:
        raise ValueError('Start must be less than end')
    
    frequency_count = 0
    
    values = []
    
    values = column_values(table, column)
    
    for val in values:
        if val >= start and val < end:
            frequency_count += 1
    
    return frequency_count
    
    
    


def histogram(table, column, nbins, xlabel, ylabel, title, filename=None):
    """Create an equal-width histogram of the given table column and number of bins.
    
    Args:
        table: The data table to use
        column: The column to obtain the value distribution
        nbins: The number of equal-width bins to use
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    
    if column not in table.columns():
        raise IndexError('Invalid column name')
    
    vals = column_values(table, column)
    
    for val in vals:
        if val == '':
            raise ValueError('Missing value detected, error')
    
    plt.figure()
    
    plt.hist(vals, density=True, bins=nbins, alpha=0.7, zorder=3, color='blue', rwidth=0.8)
    
    plt.grid(axis = 'y', color='lightgray', linestyle='-', zorder=0)


    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    

def scatter_plot_with_best_fit(table, xcolumn, ycolumn, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values that includes the "best fit" line.
    
    Args:
        table: The data table to use
        xcolumn: The column for x-values
        ycolumn: The column for y-values
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    
    if xcolumn not in table.columns() or ycolumn not in table.columns():
        raise IndexError('Invalid column name')
    
    x_vals = column_values(table, xcolumn)
    y_vals = column_values(table, ycolumn)
    
    x_mean = mean(table, xcolumn)
    y_mean = mean(table, ycolumn)
    
    plt.figure()

    plt.scatter(x_vals, y_vals, color='blue', alpha=0.5, zorder=3)
    
    plt.grid(color='lightgray', linestyle='-', zorder=0)
    
    #add regression line
    m, b = linear_regression(table, xcolumn, ycolumn)
    plt.plot(x_vals, [m * x + b for x in x_vals], color='green', linewidth=1, label='best fit')
    
    #add x and y means to the plot
    plt.axvline(x_mean, color='red', linestyle='--', label=r'$\bar{x}$', alpha=0.4)
    plt.axhline(y_mean, color='red', linestyle='--', label=r'$\bar{y}$', alpha=0.4)

    # correlation coefficient
    r_xy = round(correlation_coefficient(table, xcolumn, ycolumn), 2)
    txt = r'$r = {}$'.format(r_xy)
    plt.text(max(x_vals), max(y_vals), txt, color='red',  ha='right', va='top')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    plt.close()
    
#----------------------------------------------------------------------
# HW3
#----------------------------------------------------------------------

def distinct_values(table, column):
    """Return the unique values in the given column of the table.
    
    Args:
        table: The data table used to get column values from.
        column: The column of the table to get values from.

    Notes:
        Returns a list of unique values
    """
    
    len_test_column = [column]
    
    if len(len_test_column) > 1 or len(len_test_column) < 1:
        raise IndexError('Too many or two few columns')
    
    
    
    unique_values = []
    
    row = DataRow()
    
    for row in table:
        if row[column] not in unique_values:
            unique_values.append(row[column])
    
    return unique_values


def remove_missing(table, columns):
    """Return a new data table with rows from the given one without
    missing values.

    Args:
        table: The data table used to create the new table.
        columns: The column names to check for missing values.

    Returns:
        A new data table.

    Notes: 
        Columns must be valid.

    """
    
    return_table = DataTable(table.columns())
        
    for col in columns:

        if col not in table.columns():
            raise IndexError('Invalid column name')
        
    for row in table:

        for col in columns:

            if row[col] == '' or row[col] == None:
                break
            else:
                return_table.append(row.values())

    
    return_table = remove_duplicates(return_table)
    
    return return_table




def duplicate_instances(table):
    """Returns a table containing duplicate instances from original table.
    
    Args:
        table: The original data table to check for duplicate instances.

    """
    
    duplicate_table = DataTable(table.columns())
    table2 = table
    
    for row in table:

        duplicate = 0

        for row2 in table2:

            if row2 == row:

                duplicate = duplicate + 1
                if duplicate >= 2:
                    
                    if row2 not in duplicate_table:
                        duplicate_table.append(row2.values())

    return duplicate_table



                    
def remove_duplicates(table):
    """Remove duplicate instances from the given table.
    
    Args:
        table: The data table to remove duplicate rows from

    """
    no_duplicate_table = DataTable(table.columns())
    table2 = table
    
    for row in table:
        duplicate = 0
        for row2 in table2:
            if row2 == row:
                duplicate = duplicate + 1
                if duplicate < 2:
                    if row2 not in no_duplicate_table:
                        no_duplicate_table.append(row2.values())

    return no_duplicate_table


def partition(table, columns):
    """Partition the given table into a list containing one table per
    corresponding values in the grouping columns.
    
    Args:
        table: the table to partition
        columns: the columns to partition the table on
    """
    result = []
    final_list = []
    
    for row in table:
        value_key = []
        
        for col in columns:   # pulls all values in desired columns
            value_key.append(row[col])
        
        try: 
            if result == None or value_key not in result:  # checks if values are already in result
                result.append(value_key) # adds key to results so we don't create additional tables in the future
               
                comparison_row = DataRow(columns, value_key)
                
                new_table = DataTable(table.columns())
                for row2 in table:

                    if row2.select(columns) == comparison_row:

                        new_table.append(row2.values())
                        
                
                final_list.append(new_table)
        except TypeError:
            pass
            
    return final_list

                    



def summary_stat(table, column, function):
    """Return the result of applying the given function to a list of
    non-empty column values in the given table.

    Args:
        table: the table to compute summary stats over
        column: the column in the table to compute the statistic
        function: the function to compute the summary statistic

    Notes: 
        The given function must take a list of values, return a single
        value, and ignore empty values (denoted by the empty string)

    """
    
    all_values = []
    
    for row in table:
        value = row[column]

        # Check if the cell value is not an empty string
        if value == '':
            raise ValueError('One or more values are empty')
        else:
            # Append the non-empty value to the list
            all_values.append(value)

    # Apply the specified function to the list of non-empty values
    result = function(all_values)

    return result


def replace_missing(table, column, partition_columns, function): 
    """Replace missing values in a given table's column using the provided
     function over similar instances, where similar instances are
     those with the same values for the given partition columns.

    Args:
        table: the table to replace missing values for
        column: the coumn whose missing values are to be replaced
        partition_columns: for finding similar values
        function: function that selects value to use for missing value

    Notes: 
        Assumes there is at least one instance with a non-empty value
        in each partition

    """
    
    new_table = DataTable(table.columns())
    partitioned_tables = partition(table, partition_columns)
    
    for row in table:
        
        key_values = []
        for col in partition_columns:
            key_values.append(row[col])
            
        for i in range(len(partitioned_tables)):
            
            # this for loop creates a list of values in the partition table 
            partition_values = []
            for col in partition_columns:
                partition_values.append(partitioned_tables[i][0][col])
            
        
            if key_values == partition_values:
                if row[column] == '':

                    j = 0  # creates a variable to track the column index when comparing row and val below
                    column_index = table.columns().index(column)
                    input_row_values = []  # creates list of values to be added to the row below
                
                    for val in row.values():

                        if val == '' and column_index == j:
                            input_row_values.append(summary_stat(remove_missing(partitioned_tables[i], [column]), column, function))
                        else: 
                            input_row_values.append(val)
                        j = j + 1
                
                #creates datarow type to be appended to new table
                    final_input_row = DataRow(new_table.columns(), input_row_values)
                    new_table.append(final_input_row.values())

                else:
                    new_table.append(row.values())
        
    return new_table
    
    
             
            


def summary_stat_by_column(table, partition_column, stat_column, function):
    """Returns for each partition column value the result of the statistic
    function over the given statistics column.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups from
        stat_column: the column to compute the statistic over
        function: the statistic function to apply to the stat column

    Notes:
        Returns a list of the groups and a list of the corresponding
        statistic for that group.

    """


    input_col = [partition_column]
    partitioned_tables = partition(table, input_col)
    
    if not callable(function) or function not in (min, max):
        raise TypeError('Invalid function, please use min or max')
    
    partition_value_list = []
    stat_value_list = []

    for i in range(len(partitioned_tables)):

            partition_values = []
            for col in partition_column:
                partition_values.append(partitioned_tables[i][0][col])
            
            stat_value_list.append(summary_stat(partitioned_tables[i], stat_column, function))
            
            for row in partitioned_tables[i]:
                if row[stat_column] == summary_stat(partitioned_tables[i], stat_column, function):
                    partition_value_list.append(row[partition_column])
         
    return partition_value_list, stat_value_list


def frequencies(table, partition_column):
    """Returns for each partition column value the number of instances
    having that value.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups

    Notes:

        Returns a list of the groups and a list of the corresponding
        instance count for that group.

    """
    input_col = [partition_column]
    partitioned_tables = partition(table, input_col)
    
    partition_values = []
    frequency_values = []
    
    for i in range(len(partitioned_tables)):

            partition_values.append(partitioned_tables[i][0][partition_column])
            
            frequency_values.append(partitioned_tables[i].row_count())
            
    return partition_values, frequency_values
                
    
    


def dot_chart(xvalues, xlabel, title, filename=None):
    """Create a dot chart from given values.
    
    Args:
        xvalues: The values to display
        xlabel: The label of the x axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    yvalues = [1] * len(xvalues)
    
    plt.figure(figsize=(20, 5))  
        
    plt.plot(xvalues, yvalues, 'b.', alpha=0.2, markersize=15, zorder=3)
    
    plt.grid(color='lightgray', linestyle='-', zorder=0)

    plt.xlabel(xlabel)
    plt.title(title)
    
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    
    # Close the plot
    plt.close()

    
def pie_chart(values, labels, title, filename=None):
    """Create a pie chart from given values.
    
    Args:
        values: The values to display
        labels: The label to use for each value
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure()
    
    
    value_count = []
    for temp_label in labels:
        value_count.append(0)

    
    for val in values:
        i = 0
        for temp_label in labels:
            if val == labels[i]:
                value_count[i] = value_count[i] + 1 
            i = i + 1
    
    
    plt.pie(value_count)

    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(title)
    
    

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    plt.close()


def bar_chart(bar_values, bar_names, xlabel, ylabel, title, filename=None):
    """Create a bar chart from given values.
    
    Args:
        bar_values: The values used for each bar
        bar_labels: The label for each bar value
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure()
    
    plt.bar(bar_names, bar_values, color='b', alpha=1, zorder=3)
    
    plt.grid(axis = 'y', color='lightgray', linestyle='-', zorder=0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(range(min(bar_names), max(bar_names)+1))
    
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    plt.close()


    
def scatter_plot(xvalues, yvalues, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values.
    
    Args:
        xvalues: The x values to plot
        yvalues: The y values to plot
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    
    plt.figure()

    plt.scatter(xvalues, yvalues, color='b', alpha=0.7, zorder=3)
    
    plt.grid(color='lightgray', linestyle='-', zorder=0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()

    plt.close()


def box_plot(distributions, labels, xlabel, ylabel, title, filename=None):
    """Create a box and whisker plot from given values.
    
    Args:
        distributions: The distribution for each box
        labels: The label of each corresponding box
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    fig, ax = plt.subplots()

    value_count = []
    for temp_names in labels:
        value_count.append(0)
    
    
    for val in distributions:
        i = 0
        for temp_label in labels:
            if val == labels[i]:
                value_count[i] = value_count[i] + 1 
            i = i + 1
    
    
    ax.boxplot(distributions, labels=labels, zorder=3)
    
    plt.grid(axis = 'y', color='lightgray', linestyle='-', zorder=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
