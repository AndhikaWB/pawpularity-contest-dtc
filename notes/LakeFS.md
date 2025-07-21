## Intro

I think data versioning tools can be avoided if all of our data are strictly time series, or are immutable data. However, in my case, I only have images (recorded in a CSV/database) which doesn't have a time index, and immutability can't be guaranteed (user may delete the images).

I considered these images as streaming assets, meaning that the result may change everytime I pull the data. This makes comparing trained ML models difficult, because if the data are not the same, the model performance can't be compared to other models.

Of course I know I can just preprocess it once, save it somewhere, then always use the same preprocessed data for all models. But what if this data become out-of-date later? What if the image characteristic changes over time and I want to retrain the model using a new data? Where do I save the newly preprocessed data? What to do with the old data?

4 scenarios came to my mind:

|Scenario|Cons|Pros|
|---|---|---|
|Replace the old data with the new one on the same path|We don't have access to the old data anymore|Nothing to manage|
|Use a new path to store the new data|If the data doesn't change much, we're basically making some duplicates of the old data on the new path|Can still access the old data if needed, as long as we know the path|
|Use version control like Git for the data|Requires extra software to manage the version, since Git isn't really designed for big data|No redundant data duplicates, at least after the initial commit. Branching and rollback are also possible|
|Frequent snapshot/backup of the data|Usually involves the whole volume, not just one directory/project, so the scope is much wider. Not my job|Can be combined with the previous scenarios|

I have no problem using an extra software, and having a Git-like data versioning is pretty nice to have (though some people [disagree with this approach](https://www.reddit.com/r/MachineLearning/comments/mrb096/comment/gun8aa0)). The next problem is choosing the right tool to manage the version. Based on a quick search, I found 3 that are the most mentioned and has minimum dependency (doesn't rely on Apache softwares):

|Name|Cons|Pros|
|---|---|---|
|DVC|Since the input data can be sourced from anywhere, you basically make a duplicate on the first commit/push. Direct [import](https://dvc.org/doc/command-reference/import-url) is possible, but probably not [copy-on-write](https://en.wikipedia.org/wiki/Copy-on-write)|Flexible data storage for saving the versioned data, not just S3|
||Data repos can be anywhere (e.g. Hadoop, S3), not centralized. This may backfire if handled poorly. It relies on Git commit to track changes on the `.dvc` file/folder|Source code and data version tracking is basically one under Git. `dvc commit` should normally be followed by `git commit`|
||Data repos only contain backup of files with the hashes as their name. No commit, branch, or other reference available. To pull the right file, it relies on `.dvc` file that are saved locally|No server, no database, just a simple CLI app like `git`. No separate user management, all users use their Git account|
||No specialized web UI (like GitHub but for the data only), and probably not possible if the data repos are not centralized||
|lakeFS|Data repos must be stored on S3, though the input data can be sourced from anywhere|Allows [importing](https://docs.lakefs.io/latest/howto/import) data from existing S3 bucket without duplicating the data, like a copy-on-write approach|
||Unlike DVC, it runs a server for [tracking the metadata objects](https://docs.lakefs.io/latest/understand/how/kv) (repo, commit, etc.). Also, it only supports PostgreSQL and AWS DynamoDB for now|You can pull and commit data without cloning a whole Git project first|
||Similar to DVC, the data repos only contain files with hashes as their name, but at least it doesn't rely on local `.dvc` file/folder to track these hashes|Has web UI like GitHub for viewing the data only. Data repos can be centralized in one/more S3 buckets||
||The docs sometimes has typos and feels lacking. Feature like [ingestion](https://lakefs.io/blog/3-ways-to-add-data-to-lakefs) has existed for 3 years but hasn't been added to the [docs](https://docs.lakefs.io/latest/reference/cli/) yet||
||There are some command differences compared to Git. For example, instead of `add` it uses `upload`. I don't think it has an offline `add` mode||
|Oxen|[2x faster than DVC](https://docs.oxen.ai/features/performance), and probably uses a similar copy-on-write approach as lakeFS|Has web UI just like lakeFS, though it seems to be using it's own database, which may not be battle-tested yet|
||Has features like data labeling, model evaluation, etc. that I consider out-of-scope for a versioning tool|The commands are more similar to Git compared to lakeFS. It uses `add` instead of `upload`, etc.|
||New, doesn't seem mature enough. May check back in the future||

Note: the table above has been edited several times after taking a deeper look or trying things myself.

### My Initial Thoughts About DVC

I initially tried DVC, but it feels awkward and impossible to use in some cases. I want it to automatically track and commit new version of the data in the middle of the pipeline, before passing it to my model for testing/training. The documentation on the Python API is very lacking, there doesn't seem to be a way to add files or make a commit Pythonically. There's no way I'm calling the `dvc` CLI manually, since this task will be managed by an orchestration tool (e.g. Airflow, Prefect).

After looking online, most people that uses DVC actually use it as local orchestration tool too (e.g. before switching to Airflow on prod). They create multiple `dvc stage` where they add Python script path and file output(s) to monitor, then execute these stages using `dvc repro` or `dvc exp run` command. This is not what I want to do at all, I just want DVC to manage the data version, not as another orchestration tool.

There's this [official example](https://dvc.org/blog/automate-your-ml-pipeline-combining-airflow-dvc-and-cml-for-a-seamless-batch-scoring-experience) which shows how to run DVC with Airflow. He runs DVC on dev env to orchestrate pipeline and commit the data, but when he switch to Airflow on prod env, he only does the scoring job, after awkwardly [cloning the Git repo](https://gitlab.com/iterative.ai/cse_public/home_credit_default/-/blob/main/dags/scoring.py) and running `dvc fetch` using Bash operator on one of the Airflow tasks. Yeah sure, it can be modified somehow to do the `dvc commit` too, but cloning a whole Git repo mid pipeline is still a big no for me.

So my conclusion is `dvc commit` is designed to be run manually, and since it also depends on `git commit` to truly save the changes, it's pretty much impossible to execute it on Airflow/Prefect because there's no way to ensure the current directory is a Git project (without cloning first). Also, this assumes the worker node stays the same between tasks, which is not always the case. Not to mention the Git credentials to make the commit (which account will we use as the scape goat?).

There are people with the [same problem as me](https://github.com/iterative/dvc/discussions/5924). After trying DVC, I realized that I need a clear separation between the source code and the data, or at least a way to commit without relying on Git commit to remember the changes (and pull the data using that commit/branch reference later).

### Second Attempt using DVC

After digging more on DVC, there may be a way to use it more naturally within Python. It's not well documented, but can be used like [this](https://github.com/iterative/dvc/discussions/7379) and [this](https://stackoverflow.com/questions/76199034/dvc-checkout-without-git) reference. However, those codes have no comment at all and may be time consuming to decipher manually. In the end, I kinda give up trying it understand it, and just used this code to hijack the main CLI function of DVC:

```python
import os
import shlex
import contextlib

from dvc.cli import parse_args
from dvc.exceptions import DvcException


@contextlib.contextmanager
def change_dir(new_cwd: str = None):
    """Change directory temporarily."""

    old_cwd = os.getcwd()
    if new_cwd:
        os.chdir(new_cwd)

    try:
        yield
    finally:
        os.chdir(old_cwd)


class DvcExec():
    """Python wrapper for DVC CLI tool.

    Args:
        cwd (str, optional): Working directory for DVC.
            Defaults to None (current directory).
        quiet (bool, optional): Set log level to critical.
            Defaults to True.
    """

    def __init__(self, cwd: str = None, quiet: bool = True):
        self.cwd = cwd
        self.quiet = quiet

    def exec(self, argv: str, error: bool = False) -> int:
        """Execute DVC with specified arguments.

        Args:
            argv (str): Arguments to be passed to DVC.
            error (bool, optional): Raise DVC exception error.
                Defaults to False.

        Returns:
            int: Exit code raised by DVC.

        Raises:
            DvcException: When DVC exited with non-zero exit code.
                Only raised if `error` is set to True.
        """

        if self.cwd:
            argv = f'--cd "{self.cwd}" {argv}'
        if self.quiet:
            argv = f'-q {argv}'

        print(f'Executing "dvc {argv}"')

        args = parse_args(shlex.split(argv))
        try:
            # DVC may change cwd permanently after execution
            # The main Python script will also be affected
            # This fix will force to return to original cwd
            with change_dir(os.getcwd()):
                # Error can happen before returning stderr
                stderr = args.func(args).do_run()

            # In case no error but stderr is non-zero
            if stderr != 0:
                raise DvcException(
                    f'"dvc {argv}" returned with exit code {stderr}'
                )
        except Exception as e:
            stderr = 255
            # Prevent UnboundLocalError 
            exc = e

        if error and stderr != 0:
            raise exc

        return stderr
```

And use it like this:

```python
dvc = DvcExec(cwd = local_dir)
dvc.exec('init --no-scm -f')
dvc.exec(f'remote add test {remote_dir} -f')
dvc.exec('add *.csv --glob', error = True)
dvc.exec('add image', error = True)
dvc.exec('commit')
# <- There should be "git commit" here
dvc.exec('push -r test')
```

But since it still rely on `git commit` to track the changes made by `dvc commit`, I haven't used this as part of my pipeline yet (currently thinking how can I backup the `.dvc` files to use later without Git).

### Next Try: lakeFS

Honestly, the reason I'm not trying lakeFS first is because it needs to run a server to manage things behind the scene. I think this is kinda overkill and not portable, unless it's for reason like access control, but it should be doable using S3 role directly (which they decided not to use).

Still, I must say that lakeFS is pretty good. These sections below are written mainly for lakeFS. If I found a problem, I will also write the solution/workaround there.

## Setup

1. I assume lakeFS is already running somewhere (e.g. Docker)
2. Go to the [setup page](http://localhost:8000/setup) to set the default admin username and email
3. After creating the admin account, we will be given access key ID and secret key to login
4. There will be no repo after we login, so we must create one first to store our data
    - lakeFS will ask for the namespace (bucket location) to store that repo. This bucket must be created beforehand, lakeFS will not create it for you
    - By default, one namespace can only store one repo (lakeFS will throw an error if that namespace already has a repo). However, you can workaround this by using `s3://bucket-name/repo-id` instead of just `s3://bucket-name` to use a single bucket for multiple repos
5. Create a new non-admin account if needed to access the data repo later
6. Download the `lakectl` CLI or the `lakefs` Python library to manage lakeFS specific features (e.g. commiting data). For normal S3 operation, I recommend using AWS CLI or `boto3` library directly
7. That's it, happy experimenting!

## Tips

- When using AWS CLI or Python `boto3` library, we should use lakeFS endpoint (e.g. `localhost:8000`) rather than the original S3 endpoint
    - If we make changes to S3 without using lakeFS endpoint (e.g. upload files), these changes will not be recognized by lakeFS (even though the files are already there)
    - This is probably because of the missing metadata, which should be normally tracked by lakeFS server/endpoint
    - You can't commit this change because lakeFS doesn't detect any change (I checked this via lakeFS web UI)
- When using the lakeFS endpoint, we no longer use the bucket name. lakeFS will treat each repo as if it's a bucket. For example, instead of using `s3://bucket-name/repo-id/branch`, we should use `s3://repo-id/branch` directly
- lakeFS recommends using a separate branch for each stage of the data (e.g. dev and test). However, depending on the case, I think it's kinda okay to use just one branch with multiple CSV/parquet data (e.g. train, val, test)
- Don't track intermediate data using lakeFS (or whatever tools you use), they're useless and will only burden your storage space in the long run. Just track the final data (that are ready to be used by the ML model)

## Docs

- [Boto3 S3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
- [lakeFS with Python and boto3](https://docs.lakefs.io/latest/integrations/python/#using-boto)