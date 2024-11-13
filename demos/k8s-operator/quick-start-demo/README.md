# Kubernetes Operator via Kubebuilder


Demo based on https://book.kubebuilder.io/getting-started

## Installation

1. First make sure kubebuilder is installed
2. Then make sure you initialize the repo
    - `go mod init example.vom/m`
3. Then you can run the kubebuilder command:
    - `kubebuilder init --domain my.domain --repo my.domain/guestbook` 
4. Then create an API
    - `kubebuilder create api --group webapp --version v1 --kind Guestbook`
5. Then create a kind cluster
    - `kind create cluster --name op-demo`
        - Note: if you have issues aabout docker daemon note that it's a need to start colima via `colima start`
            - see here: https://stackoverflow.com/questions/44084846/cannot-connect-to-the-docker-daemon-on-macos

