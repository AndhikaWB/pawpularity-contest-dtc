## MLFlow Overview

Note that MLFlow 3.X has many changes so these assumptions below may no longer be true.

---

MLFlow terminology:
- **Experiment**: a parent group where you put all your run history inside it. You can also add experiment tags to group similar experiments
- **Run**: contains logged metrics, parameters, and file and model artifacts. You can compare values between different run to see which run is the best on a specific/multiple experiment(s)
    - **Metrics**: used to log values that may change over time (e.g. loss per epoch). When logging a new metric, the previous value won't be replaced so you can see the history
    - **Parameters**: unlike metrics, if you log the same parameter twice, the previous value will be replaced by the new value. However, you still can use [nested runs](https://mlflow.org/docs/latest/getting-started/quickstart-2/) as workaround
    - **Tags**: you can use it to save the developer name, framework, task type, and other useful info
    - **Artifact**: you can save model artifact and related files (e.g. optimizer state) on each run, but saving source code and dataset files are not recommended
        - Just like parameters, artifact doesn't have history/checkpoint, but you can use different artifact path or nested runs as workaround
        - EDIT: Starting from MLFlow 3.0, model artifact can have history too (it's called logged models), but this may apply to model only and not other file artifact
    - **Dataset:** input related to the model, logged using `log_input()`
        - Unlike artifact, dataset only contains metadata (e.g. source URL, column names, row count). The actual data isn't saved to MLFlow since the data can be big
        - The supported dataset format is pretty limited (only Pandas and Numpy have a great support), so most people usually rely on run tags instead
- **Model**: the result of a run (along with metrics etc. as described earlier)
    - Each run should only produce one model to avoid headache (e.g. don't save Sklearn and PyTorch model in the same run). This model can be registered under a name and then we can access it from the **model registry** tab
    - **Registered model** can be found and downloaded more easily because we can search the name, and assign **tags** and **aliases** to it
    - Multiple models can be registered under the same name, and a **version** number will be added incrementally so we can differentiate them. Think of registered model as a group of models, just like experiment and run
    - **Aliases** are bound to the model version, so if we set the same alias to a new version, the previous version won't have that alias anymore. Example aliases: king, champion, challenger
    - You can also add model **tags**. Tags from the run won't be inherited as model tags, so you will need to set the tag twice if you want to synchronize them. However, people usually use model tags to mark things like approval status rather than trivial stuff (e.g. developer name)

Best practices when using MLFlow:
- As stated earlier, one experiment should contain runs with the same task and dataset only, and each run in an experiment should only produce one model for easier comparison
    - If you use different model libraries (e.g. Sklearn and PyTorch), you should use 2 separate runs, unless you're doing ensemble, where the logged metrics are combined for all models
    - If in the big picture you need many ML models for different tasks, think of it as a separate ML project, so you should create a separate experiment, and perhaps Git repo too
- For dataset, you can pass the source URL to `log_input` along with the column schemas (names and data types)
    - For other metadata, you can save them as run tags or model tags instead
    - Saving dataset as artifact is not recommended, since each run may still use the same dataset, creating redundant copy of the data
- For source code, pass the Git repo and branch name as parameter. The Git commit hash will be tracked automatically by MLFlow as long as it's a Git directory and not a notebook file
- Use tags on each run/model like framework (e.g. Sklearn, PyTorch), task (e.g. regression, classification), and type/variant (e.g. CNN, random forest) to easily distinguish them later
- Use a separate model registry for development and production environment. You can move a model between registry (e.g. from `dev.mnist` to `prod.mnist`) by promoting the model
- Set the `registered_model_name` when calling `log_model()` during the run. This will make CI/CD easier because you don't need to find and register the model manually later
    - Currently, the `log_model()` function doesn't support adding model tags. If you want to register and add tags in one go, use `register_model()` instead (called after the run instead of during the run)
- Separate MLFlow instance per team/organization, because MLFlow has poor authentication and collaboration support. Once a user gained access to an instance, they will be granted full access to everything inside it

---

When we talk specifically about a run, ideally it should contain:
- Dataset (as tags and/or `log_input`)
- Source code (as tags)
- Metrics (e.g. loss and RMSE per epoch, and the epoch itself)
- Parameters (e.g. learning rate, max epochs, early stop patience, loss name, optimizer name)
- Model optimizer `state_dict` as file artifact (optional)
- Model input and output signature as `log_model` parameter. The signature can be different than the `log_input` dataset metadata
- Pip or Conda requirement file as `log_model` parameter (optional, can be inferred by MLFlow)
- Code paths as `log_model` parameter. Since MLFlow rely on Pickle, you should always save the model code because Pickle will use that code to load the model later. If not logged, Pickle will use the current code in the project directory which may not match the old code that was used when saving the model

If you use early stop and want to save the best model only, you should additionally:
- Save metrics history until the best epoch (e.g. as JSON/CSV artifact)
- Restore metrics at best epoch to log to MLFlow right before stopping the run

This is because latest metrics (latest epoch) doesn't equal best metrics (best epoch). MLFlow will only show latest logged metrics by default ([no min/max aggregation](https://github.com/mlflow/mlflow/issues/7790)), which can be misleading if you want to compare model with the best metrics across different runs.

However, if you want to save the models as history/checkpoint (e.g. as part of parameter tuning), you can use [nested runs](https://mlflow.org/docs/latest/getting-started/quickstart-2/) as workaround. This is because MLFlow doesn't save model history on a run, so you need multiple runs to act as history/checkpoint. 

EDIT: As of MLFlow 3, it's now possible to save multiple models in 1 run, called as logged models

---

As you can see, MLFlow is good but it has some flaws like:
- Source code and dataset management
- Authentication and collaboration

So it's best if we don't use MLFlow as an all-in-one tool to avoid future pain.