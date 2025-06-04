from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import mlflow
import os

# Define the experiment name
username = os.environ['DOMINO_STARTING_USERNAME']
experiment_name = f"random-forest-gen-{username}"

# Check if the experiment already exists
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    # Create a new experiment if it doesn't exist
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    # Use the existing experiment
    experiment_id = experiment.experiment_id

# Load the diabetes dataset
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Define the objective function for Bayesian optimization
def objective(max_depth, max_features):
    max_depth = int(max_depth)
    max_features = int(max_features)
    run_name = f"BayesOpt_max_depth_{max_depth}_max_features_{max_features}"
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.set_tag("team", "Commercial")
        mlflow.set_tag("experiment_type", "bayesian_optimization")
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth, max_features=max_features)
        
        # Train the model
        rf.fit(X_train, y_train)
        
        # Score the model
        score = rf.score(X_test, y_test)
        
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_features", max_features)
        mlflow.log_metric("accuracy", score)
        
        return score

# Define the search space
pbounds = {
    'max_depth': (2, 10),
    'max_features': (1, 10)
}

# Run Bayesian optimization
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(
    init_points=5,
    n_iter=15
)

# Print the best parameters found
print("Best parameters found: ", optimizer.max)
