# Pre-requisites

This workflow uses Celery (over Redis) for distributed parallelization of the tasks. This allows automatically scaling the work by adding or substracting workers on the fly across different clusters and simplifies the implementation of the individual workers. However, a central redis server must be available.

Noah's desktop is running a redis server that is visible from eos and ord, but firewalled on DFW. Redis can easily be run from a container on other clusters (e.g. DFW), but currently ETL is recommended on ORD.


| Cluster | Available Server|
|-|-|
| eos,ord | fb7510c-lcedt.dyn.nvidia.com|
| dfw| none |

To change the redis server modify the "CELERY_BROKER" variable in [tasks.py](./tasks.py).

## Smoke-testing redis

To test of the server is available at the expected location install the redis-cli and run
```
[I] nbrenowitz@nbrenowitz-mlt /Users/nbrenowitz
$ redis-cli -h fb7510c-lcedt.dyn.nvidia.com
fb7510c-lcedt.dyn.nvidia.com:6379> SET hello "world"
OK
fb7510c-lcedt.dyn.nvidia.com:6379> GET hello
"world"
fb7510c-lcedt.dyn.nvidia.com:6379>
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

# Curation
To start with update the csv index for the files

```python
import tasks
tasks.update_index(tasks.INPUT_ROOT)
tasks.initialize_zarr()
```
When run on an existing store `tasks.initialize_zarr()` will only initialize any new variables requested.


Launch the redis server if needed. On my desktop I do this
```
    $ docker run -d -p 6379:6379 redis
```

Queue the tasks

    python3 enqueue_curate.py

Start workers in various places

    celery -A tasks worker -c 8 --loglevel=info

To start a worker on ORD:

    sbatch submit_worker.sh

The job can be monitored using [celery flower](https://docs.celeryq.dev/en/latest/userguide/monitoring.html#flower-real-time-celery-web-monitor). You can run flower locally like this

    env FLOWER_UNAUTHENTICATED_API=true uvx --with flower --with celery[redis] celery --broker  redis://fb7510c-lcedt.dyn.nvidia.com:6379/0 flower
