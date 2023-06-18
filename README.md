# Basic Model Set up and Training (ML template)

## Load Data
* Load Data ` df = pd.read_csv(file_location)`
  * Tip: [Basic difference between Pandas, Numpy, Sklearn](https://www.quora.com/What-is-the-relationship-among-NumPy-SciPy-Pandas-and-Scikit-learn-and-when-should-I-use-each-one-of-them)

## EDA - Exploratory Data Analysis

### Clean Data
* Take a look at couple of rows and available columns - `df.head()`, `df.columns`
* Column headers - Change to lower case - `df.columns.str.lower()`
* Row values - Change to lower case
  * Get list of column names with Object types (Strings) - `list[df.dtypes[df.dtypes == 'object']].index]`
  * Loop over columns and convert to lower case, replace space with _ -  `df[col] = df[col].str.lower().str.replace(' ', '_')`
  
### Look at Data
* List of columns `df.columns`
* For each column, review number of unique values - `df[col].nunique()`, list ~5 unique values - `df[col].unique()[:5]`

### Look at Data Distribution
* Use matplotlib, seaborn to visualize data
* Histogram - `sns.histplot(df.col, bins=num)`.
  * Look for long tail distribution and look at non-tail data - `sns.histplot(df.col[df.col < somevalue], bins=num`)
  * Apply Logrithm to change actual wide apart values to smaller closer values - `np.log1p(df.col)`
* If the data now looks like a Normal Distribution (bell curve), models tend to do better.

## Pre-Process Data

### Create Train, Validation and Test DataSets
* Shuffle data and split to `df_train, df_val, df_test`
* Create target variable - `y_train = np.log1p(df_train.col.values)`, repeat for `y_val`, `y_test`
* Remove target variable from features - `del df_train[col]`, repeat for `df_val`, `df_test`

## (rephrase) Identify Relevant Features and Target Columns
* Repeat this step after pre-processing and at any other required stage
* Inspect Columns. Use correlation matrix, etc 
* Identify relevant features (X), and target (y) `data.columns, data.head(), data.describe()`  
* Identify features `features = ['col1', 'col2']`
* Get data for relevant features `X = data[features]`
* Get data for target `y = data.targetColumn`

### (rephrase) Pre-Process Data
* Identify columns with missing values and counts - `df.isnull().sum()`
* Handle Missing Values `cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()] # Get names of columns with missing values`
  * [Imputation (e.g with mean using SimpleImputer)](https://www.kaggle.com/alexisbcook/missing-values?scriptVersionId=79127568&cellId=8), 
  * [Drop missing columns (if column not relevant)](https://www.kaggle.com/alexisbcook/missing-values?scriptVersionId=79127568&cellId=6)
* Handle Categorical Values
  * Identify Categorical values `object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]`
  * [Ordinal Encoding](https://www.kaggle.com/alexisbcook/categorical-variables?scriptVersionId=79127496&cellId=12) `sklearn.preprocessing.OrdinalEncoder.fit_transform(X_train[obj_cols])`
    * May need to handle data that appears in validation, but not training.
  * [One-Hot Encoding](https://www.kaggle.com/alexisbcook/categorical-variables?scriptVersionId=79127496&cellId=14) `sklearn.preprocessing.OneHotEncoder.fit_transform(X_train[obj_cols])` 
* Handle Cyclic Features 
  * Hours of the day, days of the week, months in a year, and wind direction are all examples of features that are cyclical. 
  * Source, read: http://blog.davidkaleko.com/feature-engineering-cyclical-features.html, https://medium.com/ai%C2%B3-theory-practice-business/top-6-errors-novice-machine-learning-engineers-make-e82273d394db
  * Example: Cyclic for limited range e.g. peak daylight hours - 10AM - 3PM - [Predicting Solar Power Output using ML](https://towardsdatascience.com/predicting-solar-power-output-using-machine-learning-techniques-56e7959acb1f#:~:text=Secondly%2C%20let%E2%80%99s%20create%20cyclic%20features%20using%20month%20and%20hour%20data.%20It%20should%20be%20noted%20that%20only%20data%20between%2010%20am%20and%203%20pm%20is%20available%20which%20cuts%20out%20the%20period%20when%20the%20systems%20are%20not%20expected%20to%20generate%20power.)
  * Keep in mind that when the values are limited variations like top of the hour (24 categories), months (12), they are like categorical. You can approach with One-Hot encoding. Howver, when values are continuous, you may want to stick with cyclic approach (Source: [See comments in article](http://disq.us/p/24i2yk9))

### Split Data
* Split data into training and test sets `train_X, val_X, train_y, val_y = sklearn.model_selection.train_test_split(X, y, random_state=0)`
* #### Stratificed Sampling
  * Stratified sampling is a method to split a dataset to produce subsets containing a balanced proportion of samples for each category.
  * Stratified sampling in Machine Learning" is a quick introduction to stratified sampling.
  * Check "What is Stratified Cross-Validation in Machine Learning?" for more information about stratified cross-validation.

### Model Train/Fit
* Pick a relevant model e.g. `model = sklearn.tree.DecisionTreeRegressor(someParams) or XGBRegressor()`
* Fit with X (training data) and y (target) `model.fit(train_X, train_y)`
* Advanced - Pick appropriate model after evaluating error (next step) - e.g. some models - RandomForest, XGBoost etc.

### Predict, Validate
* Predict with validation Data `pred_y = model.predict(val_X)`
* Validate with a model quality metric, e.g. Mean Absolute Error (MAE) `sklearn.metrics.mean_absolute_error(val_y, pred_y)`

### Find Optimal Model Parameters
* Have list of parameters, feed to model, collect Error
* Chose parameters with best error (mostly least error)
* Advanced - Pick appropriate model parameters after evaluating error (earlier step) - e.g.

### Final Model
* Retrain model with full data (including validation set) and optimal parameters


## Sample Projects
(Sources as well)
* https://www.kaggle.com/thirty-days-of-ml-assignments
* Predicting Solar Power Output using ML - https://towardsdatascience.com/predicting-solar-power-output-using-machine-learning-techniques-56e7959acb1f

## Advanced Model Set up and Training (ML template)
### Use Pipelines
* Group Pre-processing steps, and other steps into individual groups. See https://www.kaggle.com/alexisbcook/pipelines
* e.g. Sklearn Pipeline

### Other topics
* Cross Validation - https://machinelearningmastery.com/k-fold-cross-validation/
