## Intro

Docker provides local Kubernetes cluster support by using Kubeadm by default. However, it's still considered heavy to use for laptop users. As an alternative, I'm using Minikube, which is the default one used on Kubernetes [getting started guide](https://kubernetes.io/docs/tutorials/kubernetes-basics/create-cluster/cluster-intro/).

See all list of official Kubernetes cluster manager [here](https://kubernetes.io/docs/tasks/tools/). There are also other non-official alternatives as well, like [MicroK8s](https://github.com/canonical/microk8s) (maintained by Canonical).

## Prerequisites
- WSL
- Docker Desktop
- Minikube

## Setup

1. Make sure all Docker related services are already running (e.g. by checking on Docker Desktop)
2. Check if kubectl is already provided by Docker or not by running `kubectl --help`. If not, download it and add to PATH manually
3. Do the same for Minikube, check if it's set up correctly by running `minikube --help`
4. Run `minikube start` to start the local Kubernetes cluster. This will download the Minikube image first if it's your first time doing it
5. That's it, run `minikube stop` to stop the cluster when you're done testing Kubernetes, and run `minikube delete` if you really want to delete the cluster