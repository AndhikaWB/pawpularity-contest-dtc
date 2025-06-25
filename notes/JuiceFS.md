## Intro

I decided to use S3 as the object storage solution because it seems to be the most supported one (e.g. compared to Hadoop). There are some players in this field, and after reading people's opinion (mostly from Reddit and Y Combinator), I came to this conclusion:

|Name|License|The good|The bad|
|---|---|---|---|
|Amazon S3|-|-|It's a service, not a software|
|MinIO|AGPL 3.0|The most popular one other than Amazon S3 and already used in production|According to [this](https://blog.min.io/from-open-source-to-free-and-open-source-minio-is-now-fully-licensed-under-gnu-agplv3), [this](https://github.com/minio/minio/issues/13308), and [this](https://opensource.stackexchange.com/a/663), all softwares that uses AGPL software must be made open-source to the users |
|Garage|AGPL 3.0|Seems good for simple usage like self-host and homelab|AGPL|
|Ceph|Mostly LGPL|Preferred by many people after the MinIO license change. Seems to be widely used in production too|More complicated to setup for demo purpose|
|SeaweedFS|Apache 2.0|Seems to be pretty well received and easy to setup, but not as popular as Ceph|People occasionally reported data loss. It may be due to user error, but I'm still worried to use it for anything serious|
|JuiceFS|Apache 2.0|Seems to be pretty popular amongst Asian companies|Rarely mentioned by individuals on Reddit, maybe because they're mostly Americans?|
|Apache Ozone|Apache 2.0|-|Seems to be a bit buggy based on others report. May check back once it matures enough|
|MooseFS|GPL 2.0|-|Haven't looked much, but not as popular as the others|

In the end, I think JuiceFS and Ceph are the winners on this list. Since JuiceFS is easier to setup, I will use it for now.

## Setup

To run JuiceFS in a Docker container, refer to [Using JuiceFS in Docker](https://juicefs.com/docs/community/juicefs_on_docker/), especially the method 2 part (using officially maintained image).