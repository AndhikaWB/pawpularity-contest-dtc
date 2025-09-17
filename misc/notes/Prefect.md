## Intro

Airflow is known to be the heaviest orchestration tool out there. I tried making Airflow more lightweight by doing these 3 things (based on this [blog post](https://datatalks.club/blog/how-to-setup-lightweight-local-version-for-airflow.html)):
- Changed from CeleryExecutor to SequentialExecutor
- Removed Redis and other Celery dependencies
- Changed from PostgreSQL to SQLite

But even with those tweaks, Airflow still randomly kill itself due to it's extensive memory usage, so I gave up using Airflow. After searching online, the 2 most popular ones seems to be Prefect and Dagster. Honestly, I hate both since I considered them as freemium softwares, but beggars can't be choosers I guess.

Dagster is definitely more popular amongst r/dataengineering folks, probably because it's data centric approach. I'm honestly impressed, but the more I looked at the docs, the more I realized that once I use Dagster, it will be hard to switch to other tools again because you're forced to use their specific methods and approaches.

Prefect on the other hand is too carefree, there's no forced rule or design principle, but this can backfire easily under inexperienced devs, especially when working on a medium-large sized team. Therefore, you may need to set your own standardized rules to be adopted by the whole team.

In the end, I chose Prefect because I won't be adopting it on my work anytime soon, and I'm still not sure about the data centric approach of Dagster (which is [also supported](https://www.prefect.io/blog/introducing-assets-from-task-to-materialize) in Prefect). Plus, it turns out Prefect has [basic auth support](https://github.com/PrefectHQ/prefect/issues/2238) on the free version, unlike Dagster which hasn't made any decision on [this issue](https://github.com/dagster-io/dagster/issues/2219) so far.

But still, even if I use Prefect, I plan to follow [Airflow principle](https://maximebeauchemin.medium.com/functional-data-engineering-a-modern-paradigm-for-batch-data-processing-2327ec32c42a) and [best practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html) in case I have to switch back from Prefect to Airflow.

---

Airflow:
- Containerize individual task
- [Secret](https://airflow.apache.org/docs/apache-airflow/stable/security/secrets/index.html) via variable, secret backend, or connection
- [Asset](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/assets.html) materialization exists, but can't be declared dynamically yet
- There can be clear code separation between DAG and the original code

Prefect:
- Containerize the whole workflow
- Secret via [secret](https://docs.prefect.io/v3/how-to-guides/configuration/store-secrets) and [block](https://docs.prefect.io/v3/concepts/blocks) (it seems that block is [basically a secret too](https://github.com/PrefectHQ/prefect/issues/14899), but I could be wrong)
- Environment variables can be configured directly to workpool or deployment via the UI (and read via `os.environ`), but Prefect [variable](https://docs.prefect.io/v3/how-to-guides/configuration/variables) also exists
- Dynamic [asset](https://docs.prefect.io/v3/how-to-guides/workflows/assets) materialization supported (but still pretty limited)


Cons of Airflow:
- Uses [stdout](https://airflow.apache.org/docs/apache-airflow-providers-docker/stable/_api/airflow/providers/docker/operators/docker/index.html) behind the scene to get the return value of containerized task. This is pretty awkward and may not be reliable ([related issue](https://stackoverflow.com/questions/63666452/airflow-docker-commands-comunicate-via-xcom))
- We can only use 1 [connection](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/connections.html) per task, so each task is practically dedicated to 1 thing only ([related issue]()). 

Cons of Prefect:
- Dynamic flow (no DAG) means you can't see the graph until you execute it
- Also, since the graph is generated "smartly", it may assume that task B has no dependency on task A if the output of task A isn't directly used by task B (e.g. task A -> modify string -> task B). This may cause an issue if parallel strategy is used
- Dynamic asset materialization is not good enough (requires you to [write custom flow](https://docs.prefect.io/v3/how-to-guides/workflows/assets#dynamic-asset-materialization) with `with_options`). This basically means your orchestration code and original code are now combined into one (unless you want to maintain 2 separate scripts with 99% similarity)