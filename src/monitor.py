def mark_best_model(metric_name: str, mlf_cfg: MLFlowConf):
    client = mlflow.MlflowClient(mlf_cfg.tracking_uri)
    model_results = client.search_model_versions(
        filter_string = f'name = \'{mlf_cfg.registered_model_name}\''
    )

    reg_models = []

    for i in model_results:
        # TODO: Check behavior on logged model and nested runs
        # For now, we assume each run can only have 1 model
        run_result = client.get_run(i.run_id)

        # Values to compare across models
        metric = run_result.data.metrics.get(metric_name, None)
        duration = run_result.info.end_time - run_result.info.start_time

        # Also calculate the model size to compare
        artifact_size = 0
        for j in client.list_artifacts(i.run_id):
            artifact_size += j.file_size if j.file_size else 0

        reg_models.append({
            'name': i.name,
            'run_id': i.run_id,
            'version': i.version,
            'tags': i.tags,
            # Sometimes a protobuf object is returned
            'aliases': list(i.aliases),
            'metric': metric,
            'duration': duration,
            'artifact_size': artifact_size
        })

    reg_models = pl.DataFrame(reg_models)

    # Sort values to get the model ranking
    # By assuming that minimum values are better
    reg_models = reg_models.sort(
        'metric', 'artifact_size',
        descending = [True, True]
    )

    # Get the best model version
    best_version = reg_models.item(0, 'version')

    # Give an alias to the best model version
    client.set_registered_model_alias(
        mlf_cfg.registered_model_name,
        alias = mlf_cfg.best_version_alias,
        version = best_version
    )

    # Return the run id of the best model version
    return reg_models.item(0, 'run_id')

def check_best(last_run_id: str, best_run_id: str):
    # If the last run is not the best then send notification
    pass