# Server E2E ASR Online

### Structure of this project:

1. master: Forward message from client(gRPC) to server(Websocket).
2. streaming_decoder: Source code of server.
3. packages: needed packages (java & miniconda) for build image.
4. utils: utils script to process some tasks.
5. exp: Scripts to test service.

### Pipeline to run project:

1. Building worker image: There are 2 methods to build:
    1.1 Building from scratch:
    ```
    $ docker build -t image_name:image_tag_cpu[,gpu] -f utils/Dockerfile_base_cpu[,gpu]
    ```
    1.2 Building from base_image:
    ```
    $ docker build -t image_name:image_tag_cpu[,gpu] -f Dockerfile_cpu[,gpu]
    ```
    or replace dockerfile in file `docker_compose.yml` and run:
    ```
    $ docker-compose build streaming-asr
    ```
2. [Optional] Building nginx image: Running as load balancing when run multiple worker containers
    ```
    $ docker-compose build nginx
    ```
3. Running service:
    3.1 Running only worker service:
    ```
    $ docker-compose up -d --force-recreate streaming-asr
    ```
    3.1 Running multiple worker and load balancing: Fix port in file `docker-compose.yml` to `- "${GRPC_PORT}"`
    ```
    $ docker-compose up -d --force-recreate --scale streaming-asr=${number_worker}
    ```
4. Testing service:
    3.1 Testing with