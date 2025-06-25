## Intro

Airflow is known to be the heaviest orchestration tool out there. I tried making Airflow more lightweight by following that does these 3 things (inspired by this [blog post](https://datatalks.club/blog/how-to-setup-lightweight-local-version-for-airflow.html)):
- Changed from CeleryExecutor to SequentialExecutor
- Removed Redis and other Celery dependencies
- Changed from PostgreSQL to SQLite

But even with those tweaks, Airflow still randomly kill itself due to it's extensive memory usage, so I gave up using Airflow. After searching online, the 2 most popular ones seems to be Prefect and Dagster. Honestly, I hate both since I considered them as freemium softwares, but beggars can't be choosers I guess.

Dagster is definitely more popular amongst r/dataengineering folks, probably because it's data centric approach. I'm honestly impressed, but the more I looked at the docs, the more I realized that once I use Dagster, it will be hard to switch to other tools again because you're forced to use their built-in functions and design principles (resulting in vendor lock-in).

Prefect on the other hand is too carefree, there's no forced rule or design principle, but this can backfire easily under inexperienced devs, especially when working on a medium-large sized team. Therefore, you may need to set your own standardized rules to be adopted by the whole team.

In the end, I chose Prefect because I won't be adopting it on my work anytime soon, and I'm still not sure about the forced data cent approach of Dagster (which Prefect recently [also implemented](https://www.prefect.io/blog/introducing-assets-from-task-to-materialize)). Plus, it turns out Prefect has [basic auth support](https://github.com/PrefectHQ/prefect/issues/2238) on the free version, unlike Dagster which hasn't made any decision on [this issue](https://github.com/dagster-io/dagster/issues/2219) so far.

But still, even if I use Prefect, I plan to follow [Airflow principle](https://maximebeauchemin.medium.com/functional-data-engineering-a-modern-paradigm-for-batch-data-processing-2327ec32c42a) and [best practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html) in case I have to switch back from Prefect to Airflow. Besides those, there are also some interesting references as well:
- Kedro [integration with Airflow](https://docs.kedro.org/en/stable/deployment/airflow.html): kinda gives an idea how task in Airflow works
- Migration [from Airflow to Prefect](https://www.prefect.io/blog/airflow-to-prefect-migration-playbook): compares Airflow concept with Prefect concept


https://airflow.apache.org/docs/apache-airflow/stable/tutorial/index.html