# Pre-requisites

This workflow uses Celery (over Redis) for distributed parallelization of the tasks. This allows automatically scaling the work by adding or substracting workers on the fly across different clusters and simplifies the implementation of the individual workers. However, you must run a separate redis server. There is much infomation about this online, but a simple way to get started is with docker

    docker run -d -p 6379:6379 redis

This will command will start a redis server in the background.

## Smoke-testing redis

To test of the server is available at the expected location install the redis-cli and run
```
$ redis-cli -h your_host
your_host:6379> SET hello "world"
OK
your_host:6379> GET hello
"world"
your_host:6379>
```
This demonstrates redis's basic key-values store capability.

## Deploying the redis server on a desktop

Follow these [docs](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/install-redis-on-linux/) to install and enable the redis service. Then, comment out any "bind" directives in `/etc/redis/redis.conf` and add a firewall exception `sudo ufw allow 6379`. Also change any lines `protected-mode yes` in the redis.conf to `protected-mode no`. This will make the redis server visible within the network, and remove password authentication.


And restart
```
sudo systemctl restart redis-server.service
```

# Regridding

Create index

    python3 create_index.py index.csv

Enqueue jobs

    python3 enqueue_regrid_jobs.py

Run a worker

    celery -A tasks worker -c 8 --loglevel=info
     
# Curation

To start with update the csv index for the files

```python
import tasks
import config
tasks.update_index(config.OUTPUT_ROOT, config.OUTPUT_PROFILE)
tasks.initialize_zarr()
```
When run on an existing store `tasks.initialize_zarr()` will only initialize any new variables requested.

Queue the tasks

    python3 enqueue_curate.py

Start workers in various places

    celery -A tasks worker -c 8 --loglevel=info

The job can be monitored using [celery flower](https://docs.celeryq.dev/en/latest/userguide/monitoring.html#flower-real-time-celery-web-monitor). You can run flower locally like this

    env FLOWER_UNAUTHENTICATED_API=true uvx --with flower --with celery[redis] celery --broker  redis://your_host:6379/0 flower
