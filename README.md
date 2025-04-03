# Machine Learning Implementation Showcase

This repository showcases implementations of various machine learning algorithms and data handling utilities. The codebase demonstrates practical applications of decision trees, k-means clustering, and other machine learning concepts.

## Project Structure

### Core Components

- `data_table.py`: A robust implementation of data table operations
  - Implements `DataRow` and `DataTable` classes for efficient data manipulation
  - Supports operations like loading/saving data, row/column manipulation, and table combinations
  - Provides type conversion and data validation capabilities

- `data_learn.py`: Implementation of machine learning algorithms
  - Decision Tree (TDIDT) implementation
  - K-means clustering
  - Naive Bayes classifier
  - K-Nearest Neighbors (KNN)
  - Random Forest implementation

- `data_eval.py`: Evaluation and testing utilities
  - Bootstrap sampling
  - Stratified holdout
  - Cross-validation
  - Performance metrics (accuracy, precision, recall)
  - Random Forest evaluation

- `data_util.py`: Utility functions for data preprocessing and analysis
  - Data cleaning and transformation
  - Statistical computations
  - Feature selection helpers

- `decision_tree.py`: Base implementation of decision tree structure
  - Tree node representation
  - Tree traversal utilities
  - Tree visualization support

## Algorithm Implementations

### Decision Tree (TDIDT)

The Top-Down Induction of Decision Trees (TDIDT) implementation follows these key steps:
1. Recursively builds a decision tree by selecting the best attribute for splitting
2. Uses information gain to determine the optimal split points
3. Handles both categorical and continuous attributes
4. Includes pruning capabilities to prevent overfitting

Example visualization using auto-mpg dataset:
![TDIDT Predict Tree](TDIDT%20Predict%20Tree.pdf)

### K-Means Clustering

The k-means implementation includes:
1. Random centroid initialization
2. Iterative centroid refinement
3. Distance-based cluster assignment
4. Total Sum of Squares (TSS) calculation for cluster quality

Example visualization using iris dataset:
![Iris Clusters](iris-clusters.png)

## Usage Examples

### Decision Tree Classification
```python
from data_table import DataTable
from data_learn import tdidt, tdidt_predict

# Load data
table = DataTable()
table.load("auto-mpg.txt")

# Build and use decision tree
tree = tdidt(table, "mpg", ["cylinders", "displacement", "horsepower", "weight"])
prediction = tdidt_predict(tree, instance)
```

### K-Means Clustering
```python
from data_table import DataTable
from data_learn import k_means

# Load data
table = DataTable()
table.load("iris.txt")

# Perform clustering
centroids = k_means(table, initial_centroids, ["sepal_length", "sepal_width"])
```

## Dataset Examples

The repository includes several datasets for testing and demonstration:
- `auto-mpg.txt`: Vehicle characteristics and fuel efficiency
- `iris.txt`: Classic iris flower measurements
- `titanic.txt`: Passenger information from the Titanic
- `student-stress.txt`: Student stress level indicators

## Technical Details

### Data Table Implementation
- Efficient row and column operations
- Support for various data types
- Memory-efficient data storage
- Flexible data loading and saving

### Machine Learning Algorithms
- Modular design for easy extension
- Comprehensive evaluation metrics
- Support for both supervised and unsupervised learning
- Cross-validation and bootstrap sampling for robust evaluation

## Future Improvements

Potential areas for enhancement:
1. Parallel processing for large datasets
2. Additional algorithm implementations
3. Enhanced visualization capabilities
4. Performance optimizations for large-scale data 
