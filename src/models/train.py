import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifire
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , recall_score



def train_model(df : pd.Dataframe , targt_col : str):
    """
    train a XGBoost model and mlflow
    """
    X = df.drop(columns = [target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2 , random_state = 42
    )


    model = XGBClassifire(
        n_estimators =300,
        learning_rate =0.1,
        max_depth = 5,
        random_state = 42,
        n_jobs = -1,
        eval_metrics = "logloss"
    )

    with mlflow.start():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test , pred)
        rec = recall_score(y_test , pred)



        mlflow.log_param("n_estimators" , 300 )
        mlflow.log_param("accuracy" , acc)
        mlflow.log_param("recall", rec)
        mlflow.xgboost.log_param(model , "model")

        train_ds = mlflow.data.from_pandas(df , source = "training_data")
        mlflow.log_input(drain_ds , context = "training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")
