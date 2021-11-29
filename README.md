## Basic Model Set up and Training (ML template)

### Load Data
* Load Data ` data = pd.read_csv(file_location)`
  * Tip: [Basic difference between Pandas, Numpy, Sklearn](https://www.quora.com/What-is-the-relationship-among-NumPy-SciPy-Pandas-and-Scikit-learn-and-when-should-I-use-each-one-of-them)

### Identify Relevant Features (Columns)
* Inspect Columns
* Identify relevant features (X), and target (y) `data.columns, data.head(), data.describe()`  
* Identify features `features = ['col1', 'col2']`
* Get data for relevant features `X = data[features]`
* Get data for target `y = data.targetColumn`

### Split Data
* Split data into training and test sets `train_X, val_X, train_y, val_y = sklearn.model_selection.train_test_split(X, y, random_state=0)`

### Pre-Process Data
* Handle Missing Values `cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()] # Get names of columns with missing values`
  * [Imputation (e.g with mean using SimpleImputer)](https://www.kaggle.com/alexisbcook/missing-values?scriptVersionId=79127568&cellId=8), 
  * [Drop missing columns (if column not relevant)](https://www.kaggle.com/alexisbcook/missing-values?scriptVersionId=79127568&cellId=6)
* Handle Categorical Values
  * Identify Categorical values `obj_cols = X_train.select_dtypes..`
  * Use Ordinal/One-Hot Encoding
    * [Ordinal Encoding](https://www.kaggle.com/alexisbcook/categorical-variables?scriptVersionId=79127496&cellId=12) `sklearn.preprocessing.OrdinalEncoder.fit_transform(X_train[obj_cols])`
    * [One-Hot Encoding](https://www.kaggle.com/alexisbcook/categorical-variables?scriptVersionId=79127496&cellId=14) `sklearn.preprocessing.OneHotEncoder.fit_transform(X_train[obj_cols])` 

### Model Train/Fit
* Pick a relevant model e.g. `model = sklearn.tree.DecisionTreeRegressor(someParams)`
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

---
###### Sources 
- https://www.kaggle.com/thirty-days-of-ml-assignments (best to start and complete)
- https://machinelearningmastery.com/blog/
