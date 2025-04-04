# Data Science Algorithms Implementation

This repository showcases implementations of various machine learning algorithms and data handling utilities developed for CPSC 322: Data Science Algorithms. The codebase demonstrates practical applications of various machine learning concepts, with a focus on explicit implementation rather than relying on high-level libraries.

## Project Structure

### Core Files

1. `data_table.py`
   - Purpose: Core data structure implementation
   - Similar to pandas/numpy but with explicit implementation
   - Key components:
     - `DataRow` class for individual row representation
     - `DataTable` class for table operations
     - Data loading/saving functionality
     - Type conversion and validation
   - Example Usage:
   ```python
   # Create and load a table
   table = DataTable(['mpg', 'cylinders', 'displacement'])
   table.load('auto-mpg.txt')
   
   # Append a new row
   new_row = {'mpg': 25.0, 'cylinders': 4, 'displacement': 120.0}
   table.append(new_row)
   
   # Access rows and columns
   first_row = table[0]
   mpg_values = [row['mpg'] for row in table]
   ```

2. `data_util.py`
   - Purpose: Data preprocessing utilities
   - Includes:
     - Data cleaning functions
     - Statistical computations
     - Feature selection helpers
     - Data normalization and discretization

3. `data_learn.py`
   - Purpose: Machine learning algorithm implementations
   - Contains:
     - K-Nearest Neighbors (KNN)
     - Naive Bayes classifier
     - TDIDT (Top-Down Induction of Decision Trees)
     - Random Forest implementation
     - K-means clustering

4. `data_eval.py`
   - Purpose: Model evaluation and testing
   - Features:
     - Sampling methods (bootstrap, stratified holdout)
     - Cross-validation implementations
     - Confusion matrix generation
     - Performance metrics calculation

5. `decision_tree.py`
   - Purpose: Base decision tree structure/classes
   - ***Note: This file was provided by the course professor
   - Provides:
     - Tree node structure
     - Basic tree operations
     - Tree traversal utilities
     - Visualization support

## Data Processing

### 1. Data Normalization
- Purpose: Ensures features contribute equally to distance calculations in kNN algorithm
- Logic: Scales numerical features to a standard range (typically 0-1)
- Implementation:
```python
def normalize(table, column):
    """Normalizes values in a column to range [0,1]."""
    values = [row[column] for row in table]
    min_val = min(values)
    max_val = max(values)
    
    # Avoid division by zero
    if max_val == min_val:
        return
    
    # Normalize each value
    for row in table:
        row[column] = (row[column] - min_val) / (max_val - min_val)
```
- Example Usage:
```python
# Normalize displacement and weight for auto-mpg dataset
normalize(auto, 'disp')
normalize(auto, 'weight')
```

### 2. Sampling Methods

Sampling methods used for evaluating machine learning models by splitting data into training and testing sets. Training data is used to build the model, while testing data evaluates its performance on unseen examples.

#### 1. Bootstrap Sampling
- Logic: Creates training set by sampling with replacement, remaining samples form testing set
- Advantage: Useful for small datasets, creates multiple training sets
- Implementation:
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

#### 2. Stratified Holdout
- Logic: Maintains class distribution in both training and testing sets
- Advantage: Preserves data distribution, crucial for imbalanced datasets 
- Implementation:
```python
def stratified_holdout(table, label_col, test_size):
    """Creates stratified training and testing sets."""
    # Group rows by class label
    class_groups = {}
    for row in table:
        label = row[label_col]
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(row)
    
    # Create stratified split
    training_set = DataTable(table.columns())
    testing_set = DataTable(table.columns())
    
    for label, rows in class_groups.items():
        # Calculate split size for this class
        class_test_size = int(len(rows) * test_size)
        # Randomly select test samples
        test_indices = random.sample(range(len(rows)), class_test_size)
        
        # Split into train and test
        for i, row in enumerate(rows):
            if i in test_indices:
                testing_set.append(row.values())
            else:
                training_set.append(row.values())
    
    return (training_set, testing_set)
```

#### 3. Cross-Validation
- Logic: Splits data into k folds, uses each fold as testing set once
- Advantage: More robust evaluation, uses all data for both training and testing
- Implementation:
```python
def cross_validation(table, k, label_col):
    """Performs k-fold cross-validation."""
    folds = []
    fold_size = table.row_count() // k
    
    # Create k folds
    for i in range(k):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k-1 else table.row_count()
        fold = DataTable(table.columns())
        for row in table[start_idx:end_idx]:
            fold.append(row.values())
        folds.append(fold)
    
    # Perform k-fold validation
    results = []
    for i in range(k):
        # Use fold i as testing set
        test_set = folds[i]
        # Combine remaining folds for training
        train_set = DataTable(table.columns())
        for j in range(k):
            if j != i:
                for row in folds[j]:
                    train_set.append(row.values())
        results.append((train_set, test_set))
    
    return results
```

## Machine Learning Algorithms

### 1. K-Nearest Neighbors (KNN)
- Logic: Classifies instances based on majority vote of k nearest neighbors
- Key Components:
  - Distance calculation between instances
  - K parameter selection
  - Majority voting mechanism
- Implementation:
```python
def knn_classify(train_set, test_instance, k, label_col, features):
    """Classify instance using k-nearest neighbors."""
    # Calculate distances to all training instances
    distances = []
    for train_instance in train_set:
        dist = calculate_distance(test_instance, train_instance, features)
        distances.append((dist, train_instance[label_col]))
    
    # Get k nearest neighbors
    distances.sort()
    k_nearest = distances[:k]
    
    # Majority vote
    votes = {}
    for _, label in k_nearest:
        votes[label] = votes.get(label, 0) + 1
    
    return max(votes.items(), key=lambda x: x[1])[0]
```

- Results (Auto MPG Dataset):
  - Preprocessing:
    - Discretized MPG into 4 bins
    - Normalized displacement and weight
    - Removed missing values
  - Evaluation:
    ```python
    result_confusion_matrix = knn_stratified(auto, 10, 'mpg', majority_vote, 7, ['weight','disp'], [])
    ```
  - Class Labels:
    - 1: Low MPG (0-18 mpg)
    - 2: Medium MPG (18-24 mpg)
    - 3: High MPG (24+ mpg)
  - Performance:
    - Class 1: High accuracy (0.890) and precision (0.905)
    - Class 2: Moderate performance with good recall (0.878)
    - Class 3: Struggles with predictions (0 precision/recall)

### 2. Naive Bayes
- Logic: Uses Bayes' theorem with feature independence assumption
- Key Components:
  - Prior probability calculation
  - Likelihood estimation
  - Laplace smoothing
- Implementation:
```python
def naive_bayes_classify(train_set, test_instance, label_col, features):
    """Classify instance using Naive Bayes."""
    # Calculate prior probabilities
    priors = calculate_priors(train_set, label_col)
    
    # Calculate likelihoods
    likelihoods = {}
    for label in priors:
        likelihoods[label] = 1.0
        for feature in features:
            p = calculate_likelihood(train_set, test_instance, feature, label)
            likelihoods[label] *= p
    
    # Calculate posterior probabilities
    posteriors = {}
    for label in priors:
        posteriors[label] = priors[label] * likelihoods[label]
    
    return max(posteriors.items(), key=lambda x: x[1])[0]
```

- Results (Auto MPG Dataset):
  - Preprocessing: Same as KNN
  - Evaluation:
    ```python
    result_confusion_matrix = naive_bayes_stratified(auto, 10, 'mpg', ['weight','disp'], [])
    ```
  - Class Labels: Same as KNN
  - Performance:
    - Class 1: Strong performance (0.889 accuracy, 0.912 precision)
    - Class 2: Balanced performance (0.804 accuracy)
    - Class 3: Better than KNN but still challenging (0.422 precision)

### 3. Decision Tree (TDIDT)
- Logic: Recursively builds tree by selecting best attribute for splitting
- Key Components:
  - Entropy calculation for measuring information gain
  - Attribute selection based on maximum information gain
  - Tree pruning to prevent overfitting
- Implementation (Entropy Calculation):
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
*Note: Full TDIDT implementation is available in data_learn.py*

- Results (Auto MPG Dataset):
  - Preprocessing: Same as KNN
  - Visualization: [TDIDT Predict Tree](TDIDT%20Predict%20Tree.pdf)

### 4. Random Forest
- Logic: Ensemble of decision trees with random feature selection
- Key Components:
  - Multiple decision trees
  - Random feature subset selection
  - Majority voting for final prediction
- Implementation:
```python
def random_forest(train_set, test_instance, n_trees, features, label_col):
    """Classify instance using random forest."""
    predictions = []
    for _ in range(n_trees):
        # Select random feature subset
        selected_features = random.sample(features, int(len(features) ** 0.5))
        # Build tree and predict
        tree = tdidt(train_set, selected_features, label_col)
        pred = predict(tree, test_instance)
        predictions.append(pred)
    
    # Majority vote
    return max(set(predictions), key=predictions.count)
```

### 5. K-Means Clustering
- Logic: Groups similar data points into k clusters by minimizing within-cluster variance
- Key Components:
  - Centroid initialization
  - Distance-based cluster assignment
  - Iterative centroid refinement (recursively edit centroid)
  - Total Sum of Squares (TSS) calculation
- Implementation:
```python
def k_means(table, initial_centroids, features):
    """Performs k-means clustering on the given data."""
    k = len(initial_centroids)
    centroids = initial_centroids
    old_centroids = None
    
    while centroids != old_centroids:
        # Assign points to nearest centroid
        clusters = [[] for _ in range(k)]
        for row in table:
            min_dist = float('inf')
            nearest_centroid = 0
            for i, centroid in enumerate(centroids):
                dist = calculate_distance(row, centroid, features)
                if dist < min_dist:
                    min_dist = dist
                    nearest_centroid = i
            clusters[nearest_centroid].append(row)
        
        # Update centroids
        old_centroids = centroids
        centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = calculate_centroid(cluster, features)
                centroids.append(new_centroid)
    
    return centroids
```
- Results (iris dataset):
  - Visualization: ![Iris Clusters](iris-clusters.png)
  - Shows clear separation of iris species based on petal and sepal measurements

## Sample Algorithm Comparison

### Auto MPG Dataset Analysis
The dataset classifies vehicles into three MPG categories:
- Class 1 (Low MPG): 0-18 mpg
- Class 2 (Medium MPG): 18-24 mpg
- Class 3 (High MPG): 24+ mpg

Key Findings:
1. KNN Performance:
   - Excels at identifying low MPG vehicles (Class 1)
   - Struggles with high MPG vehicles (Class 3)
   - Moderate performance on medium MPG vehicles
   - Best suited for identifying fuel-inefficient vehicles

2. Naive Bayes Performance:
   - More balanced across all classes
   - Successfully identifies some high MPG vehicles
   - Better at handling medium MPG vehicles
   - More reliable for general classification tasks

3. Overall Comparison:
   - KNN: Better for binary classification (efficient vs inefficient)
   - Naive Bayes: Better for multi-class classification
   - Both struggle with medium MPG vehicles due to overlap in features
   - Feature importance: Weight and displacement are strong predictors

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
