## Intro

I think data versioning tools can be avoided if all of our data are strictly time series data. However, in my case, I only have images (recorded in a CSV file) which doesn't have a time index.

I considered these images as streaming assets (and the CSV as database), meaning that the result may change everytime I pull the data. This makes comparing models difficult because each model may use a different set of data, so they can't be compared to each other.

Of course I know I can just preprocess it once, save it somewhere, then always use the same preprocessed data for all models. But what if this data become out-of-date later? What if the image characteristic changes over time and I want to retrain the model using a new data? Where do I save the newly preprocessed data then?

4 scenarios came to my mind:

|Scenario|Cons|Pros|
|---|---|---|
|Replace the old data with the new one on the same path|We don't have access to the old data anymore|Nothing to manage|
|Use a new path to store the new data|If the data doesn't change much, we're basically making duplicates of similar data on each new path|Can still access the old data easily if needed|
|Use version control like Git for the data|Requires extra software to manage the version, since Git isn't really designed for big data|No redundant data duplicates, at least after the initial commit. Branching and rollback are also possible|
|Frequent snapshot/backup of the data|Usually involves the whole volume, not just one directory/project, so the scope is much wider. Definitely not my job|Can be combined with the previous scenarios|

I have no problem using extra software to manage the data version, and having a Git-like data versioning would be cool (though some people [disagree with this approach](https://www.reddit.com/r/MachineLearning/comments/mrb096/comment/gun8aa0)). The next problem is choosing the software/tool to manage the version. Based on a quick search, I found 3 that are the most mentioned and semi-standalone (doesn't rely on Apache Spark or whatever):

|Name|Cons|Pros|
|---|---|---|
|DVC|Since the input data can be sourced from anywhere, you basically make a duplicate on the first commit and push. Direct [import](https://dvc.org/doc/command-reference/import-url) is possible, but most likely not [copy-on-write](https://en.wikipedia.org/wiki/Copy-on-write)|Flexible data storage for saving the versioned data, not just S3|
||Data repos can be anywhere (e.g. Hadoop, S3), not centralized. This may backfire if handled poorly. It relies on Git commit to track changes on the `.dvc` folder/file|Source code and data version tracking is basically one under Git. `dvc commit` should normally be followed by `git commit`|
||No specialized web UI (like GitHub but for the data only), and probably not possible if the data repos are not centralized|No server, no database, just a simple CLI app like `git`. No separate user management, user is Git user|
|lakeFS|Data repos must be stored on S3, though the input data can be sourced from anywhere|Allows [importing](https://docs.lakefs.io/latest/howto/import) data from existing S3 bucket without duplicating the data, like a copy-on-write approach|
||Unlike DVC, it runs a server for [tracking the metadata objects](https://docs.lakefs.io/latest/understand/how/kv) (repo, commit, etc.), and only supports PostgreSQL and AWS DynamoDB for now|You can pull and commit data without cloning a Git project first. There's no mention of this for DVC|
||The docs sometimes has typos and feels lacking. Feature like [ingestion](https://lakefs.io/blog/3-ways-to-add-data-to-lakefs) exists for years but doesn't exist on the [docs](https://docs.lakefs.io/latest/reference/cli/)|Has web UI like GitHub for viewing the data only. Data repos can be centralized in one/more S3 buckets|
||Doesn't seem to have an `add` command to make a simple change, meaning that we need to clone/mount the whole data repo to make changes, then `commit` directly||
|Oxen|Fast, [compared to DVC](https://docs.oxen.ai/features/performance), and probably uses a similar copy-on-write approach as lakeFS|Has web UI just like lakeFS, though it seems to be using a home-baked database|
||Has features like data labeling, model evaluation, etc. that I consider out-of-scope for a versioning tool|Possible to make new changes without cloning the whole repo|
||New, doesn't seem mature enough. May check back in the future||

Note: the table above has been edited by me several times after taking a deeper look or trying things myself.

### My Initial Thoughts on DVC

I initially tried DVC, but it feels awkward and impossible to use in some cases. I want it to automatically track and commit new version of the data in the middle of the pipeline, before passing it to my model for testing/training. The documentation on the Python API is very lacking, there doesn't seem to be a way to add file or make commit. There's no way I'm calling the `dvc` CLI manually, since this task will be managed by orchestration tool (e.g. Airflow, Prefect).

After looking online, most people that uses DVC actually use it as a local orchestration tool too (e.g. before switching to prod). We can create multiple `dvc stage` where we add the Python script files and the file outputs to monitor, then execute theses stage using the `dvc repro` or `dvc exp run` command. This is not what I want to do at all, I just want DVC to manage the data version, not as another orchestration tool.

There's this [official example](https://dvc.org/blog/automate-your-ml-pipeline-combining-airflow-dvc-and-cml-for-a-seamless-batch-scoring-experience) which shows how to run DVC with Airflow. He runs DVC on dev env to orchestrate pipeline and commit the data, but when he switch to Airflow on prod env, he only does the scoring job, after awkwardly [cloning the Git repo](https://gitlab.com/iterative.ai/cse_public/home_credit_default/-/blob/main/dags/scoring.py) and running `dvc fetch` using Bash operator on one of the Airflow tasks. Yeah sure, it can be modified somehow to do the `dvc commit` too, but cloning a whole Git repo mid pipeline is a big no for me.

So my initial conclusion is DVC commit is designed to be run manually, and since it depends on Git commit too to truly save the changes, it's pretty much impossible to execute it using Airflow/Prefect because there's no way to ensure the current directory is a Git project (without cloning first). Also, this assumes the worker node stays the same between tasks, which is not always the case. Not to mention the Git credentials that must be supplied mid pipeline too.

There are people with the [same problem as me](https://github.com/iterative/dvc/discussions/5924). After trying DVC, I realized that I need a clear separation between the source code and the data, or at least a separate way to pull, commit, and push without cloning the source code repo first.

**EDIT:** After digging more on DVC, there may be a way to pull, commit, and push data on prod using the Python API (without cloning the Git repo). It's not well documented, but can be used based on [this](https://github.com/iterative/dvc/discussions/7379) and [this](https://stackoverflow.com/questions/76199034/dvc-checkout-without-git) reference. However, those codes have no comment at all and too time consuming for me to decipher manually.

### Next Try: lakeFS

Honestly, the reason I'm not trying lakeFS first is because it needs to run a server to manage things behind the scene. I think this is kinda overkill and not portable, unless it's for reason like access control, but it should be doable using S3 role directly (which they decided not to use).

Still, I must say that lakeFS is pretty good. There are some sections dedicated to lakeFS below. If I found a problem, I will also write the solution/workaround there.

## Setup

lakeFS recommends using a separate branch for each stage of the data (e.g. dev and test). DVC can even use a more extreme approach where you can stage the data on each step of the pipeline.

Some people seem to [disagree with this approach](https://www.reddit.com/r/MachineLearning/comments/mrb096/comment/gun8aa0), especially if there are many intermediate steps. I kinda agree with them too, so I decided to just version the final data, which is important for ML model reproducibility.

1. Go to http://localhost:8000/setup to set the default admin username and email
2. After creating the admin account, we will be given access key ID and secret key to login
3. There will be no repo after we login, so we must create one first to store our data
    - lakeFS will ask for the namespace (bucket location) to store that repo. This bucket must be created beforehand, lakeFS will not create it for you
    - By default, one namespace can only store one repo (lakeFS will error if that namespace already has a repo). However, you can workaround this by using `s3://bucket-name/repo-name` instead of just `s3://bucket-name` to use a single bucket for multiple repos
4. Download `lakectl` CLI from their GitHub release, or use the Python client (`pip install lakefs`)
5. That's it, happy experimenting!