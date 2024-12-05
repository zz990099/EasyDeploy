# EasyDeploy Enviroment Setup

We use docker to setup enviroment. 

## Scripts

- Just run docker container building script.
    ```bash
    cd docker
    bash build_docker.sh --platform=nvidia_gpu # rk3588/jetson
    ```

- Run script to get into container.
    ```bash
    bash docker/into_docker.sh
    ```

## Notes

- `docker` is needed on your machine. You could use scripts to easily install docker.
    ```bash
    wget http://fishros.com/install -O fishros && . fishros
    ```

- On `jetson` platform, `docker` is pre-installed with jetpack. There is no need to install docker again.

- On `nvidia_gpu` platform, `nvidia-container-runtime` is needed by docker to use gpu and cuda. Please make sure it is installed and configured with docker daemon. 
    ```bash
    cat /etc/docker/daemon.json
    # `nvidia` should be in runtimes list
    ```
- Downloading packages from `github` during docker image building may break up. Use `github` proxy from repo [gh-proxy](https://github.com/hunshcn/gh-proxy) to speed it up.
