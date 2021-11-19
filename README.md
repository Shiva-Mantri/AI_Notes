## Basic Model Set up and Training
### Set Up Data
* Load Data ` data = pd.read_csv(file_location)`
* Inspect Columns, identify relevant features (X), and target (y) `data.columns, data.head(), data.describe()`
* Clean Data, e.g. remove nulls, etc  `data.dropna(axis=0)`
* Identify features `features = ['col1', 'col2']`
* Get data for relevant features `X = data[features]`
* Get data for target `y = data.targetColumn`
* Split data into training and test sets `train_X, test_X, train_y, val_y = sklearn.model_selection.train_test_split(X, y, random_state=0)`

### Model Train/Fit
* Pick a relevant model e.g. `model = sklearn.tree.DecisionTreeRegressor(someParams)`
* Fit with X (training data) and y (target) `model.fit(train_X, train_y)`

### Predict and validate
* Predict with Test Data `pred_y = model.predict(test_X)`
* Validate with a model quality metric, e.g. Mean Absolute Error `sklearn.metrics.mean_absolute_error(test_y, pred_y)`
