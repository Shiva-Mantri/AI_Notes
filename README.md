## Basic Model Set up and Training (ML template)

Sources 
- https://www.kaggle.com/thirty-days-of-ml-assignments

### Set Up Data
* Load Data ` data = pd.read_csv(file_location)`
* Inspect Columns, identify relevant features (X), and target (y) `data.columns, data.head(), data.describe()`
* Clean Data, e.g. remove nulls, etc  `data.dropna(axis=0)`
* Identify features `features = ['col1', 'col2']`
* Get data for relevant features `X = data[features]`
* Get data for target `y = data.targetColumn`
* Split data into training and test sets `train_X, val_X, train_y, val_y = sklearn.model_selection.train_test_split(X, y, random_state=0)`

### Model Train/Fit
* Pick a relevant model e.g. `model = sklearn.tree.DecisionTreeRegressor(someParams)`. Adv - Picking a model
* Fit with X (training data) and y (target) `model.fit(train_X, train_y)`

### Predict, Validate
* Predict with validation Data `pred_y = model.predict(val_X)`
* Validate with a model quality metric, e.g. Mean Absolute Error `sklearn.metrics.mean_absolute_error(val_y, pred_y)`

### Find Optimal Model Parameters
* Have list of parameters, feed to model, collect Error
* Chose parameters with best error (mostly least error)

### Final Model
* Retrain model with full data (including validation set) and optimal parameters

## Picking a Model
Some of the model types
### Ensemble methods
https://scikit-learn.org/stable/modules/ensemble.html

#### RandomForest
The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. If you keep modeling, you can learn more models with even better performance, but many of those are sensitive to getting the right parameters.
