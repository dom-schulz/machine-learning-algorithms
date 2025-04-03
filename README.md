# Data Science Algorithms Implementation

This repository showcases implementations of various machine learning algorithms and data handling utilities developed for CPSC 322: Data Science Algorithms. The codebase demonstrates practical applications of decision trees, k-means clustering, and other machine learning concepts, with a focus on explicit implementation rather than relying on high-level libraries.

## Project Structure

### Core Components

- `data_table.py`: A robust implementation of data table operations
  - Similar to pandas/numpy but with explicit implementation for educational purposes
  - Implements `DataRow` and `DataTable` classes for efficient data manipulation
  - Supports operations like loading/saving data, row/column manipulation, and table combinations
  - Provides type conversion and data validation capabilities
  - Example usage:
  ```python
  table = DataTable(['mpg', 'cylinders', 'displacement'])
  table.load('auto-mpg.txt')
  ```

## Sampling Methods

The `data_eval.py` module implements several sampling techniques for machine learning evaluation:

### Bootstrap Sampling
```python
def bootstrap(table):
    """Creates training and testing sets using bootstrap method."""
    total_rows = table.row_count()
    training_set = DataTable(table.columns())
    testing_set = DataTable(table.columns())
    
    # Sample with replacement for training set
    while training_set.row_count() < total_rows:
        rand_index = randint(0, total_rows - 1)
        training_set.append(table[rand_index].values())
    
    # Remaining rows form testing set
    for row in table:
        if row.values() not in training_set:
            testing_set.append(row.values())
    
    return (training_set, testing_set)
```

### Stratified Holdout
- Maintains class distribution in both training and testing sets
- Particularly useful for imbalanced datasets
- Implementation ensures proportional representation of classes

### Cross-Validation
- Implements k-fold cross-validation
- Supports both standard and stratified versions
- Useful for robust model evaluation

## Performance Evaluation

### Confusion Matrix Implementation
Example using auto-mpg dataset:
```python
# KNN Results
Actual     1    2    3
--------  ---  ---  ---
       1  265   58    1
       2   31  184   31
       3    0   18   26

# Naive Bayes Results
Actual     1    2    3
--------  ---  ---  ---
       1  429   56    1
       2   42  261   66
       3    0   15   51
```

Performance Metrics:
- Accuracy: Measures overall correct predictions
- Precision: Ratio of true positives to all positive predictions
- Recall: Ratio of true positives to all actual positives
- F-measure: Harmonic mean of precision and recall

## Machine Learning Algorithms

### Decision Tree (TDIDT)

The Top-Down Induction of Decision Trees implementation uses entropy and information gain:

```python
def entropy(table, label_col):
    """Calculate entropy for a given label column."""
    total = table.row_count()
    if total == 0:
        return 0
    
    # Calculate probability distribution
    counts = {}
    for row in table:
        label = row[label_col]
        counts[label] = counts.get(label, 0) + 1
    
    # Calculate entropy
    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    
    return entropy
```

Key features:
1. Information gain calculation for attribute selection
2. Handling of both categorical and continuous attributes
3. Pruning capabilities to prevent overfitting

Example visualization using auto-mpg dataset:
![TDIDT Predict Tree](TDIDT%20Predict%20Tree.pdf)

### K-Nearest Neighbors (KNN)

Implementation features:
1. Distance-based classification
2. Support for both numerical and categorical attributes
3. Majority voting for classification
4. Customizable k parameter

Example results from auto-mpg dataset:
```python
# KNN with k=7, 10-fold cross-validation
Accuracy of Class Label 1: 0.890
Precision of Class Label 1: 0.905
Recall of Class Label 1: 0.884
F Measure of Class Label 1: 0.895
```

### Naive Bayes

Implementation includes:
1. Probability estimation for each class
2. Feature independence assumption
3. Laplace smoothing for handling zero probabilities

Example results from auto-mpg dataset:
```python
# Naive Bayes with 10-fold cross-validation
Accuracy of Class Label 1: 0.889
Precision of Class Label 1: 0.912
Recall of Class Label 1: 0.875
F Measure of Class Label 1: 0.893
```

### Algorithm Comparison

Based on the auto-mpg dataset results:
1. KNN performs better for class label 1 with higher precision
2. Naive Bayes shows better performance for class label 3
3. Both algorithms struggle with class label 2, though Naive Bayes shows slightly better recall
4. Overall, Naive Bayes provides more balanced performance across classes

## Dataset Examples

The repository includes several datasets for testing and demonstration:
- `auto-mpg.txt`: Vehicle characteristics and fuel efficiency
- `iris.txt`: Classic iris flower measurements
- `titanic.txt`: Passenger information from the Titanic
- `student-stress.txt`: Student stress level indicators

## Technical Implementation Details

### Data Processing
- Efficient matrix operations for large datasets
- Memory-optimized data structures
- Type-safe data handling
- Robust error handling and validation

### Algorithm Optimization
- Vectorized operations where possible
- Efficient data structures for nearest neighbor search
- Optimized probability calculations for Naive Bayes
- Smart pruning strategies for decision trees

## Future Improvements

Potential areas for enhancement:
1. Parallel processing for large datasets
2. Additional algorithm implementations
3. Enhanced visualization capabilities
4. Performance optimizations for large-scale data 
