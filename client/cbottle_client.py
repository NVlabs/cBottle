# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import aiohttp
import asyncio
from typing import Iterable, AsyncIterator, Literal
from urllib.parse import urljoin
import xarray
import random
from zarr.core.buffer import default_buffer_prototype, Buffer, BufferPrototype
from zarr.core.common import BytesLike
from zarr.abc.store import (
    Store,
    ByteRequest,
)


class ConsulLoadBalancedStore(Store):
    def __init__(
        self,
        service_name: str,
        consul_host: str = "localhost",
        consul_port: int = 8500,
        prefix: str = "",
        routing_strategy: Literal["hash", "round-robin"] = "hash",
    ):
        super().__init__(read_only=True)
        self.service_name = service_name
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.prefix = prefix.rstrip("/")  # Remove trailing slash if present
        self.index = random.randint(0, 1000)
        self.instances = []
        self.lock = asyncio.Lock()
        self.session = None
        self.routing_strategy = routing_strategy

    async def _open(self):
        await super()._open()
        if self.session is None:
            self.session = aiohttp.ClientSession()
        await self._refresh_instances()

    async def _refresh_instances(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

        url = f"http://{self.consul_host}:{self.consul_port}/v1/health/service/{self.service_name}?passing=true"
        async with self.session.get(url) as resp:
            resp.raise_for_status()
            services = await resp.json()

        self.instances = [
            f"http://{s['Service']['Address']}:{s['Service']['Port']}" for s in services
        ]
        if not self.instances:
            raise RuntimeError("No healthy services found")

    async def _get_instance(self, path: str) -> str:
        async with self.lock:
            if not self.instances:
                await self._refresh_instances()
            if self.routing_strategy == "hash":
                instance = self.instances[hash(path) % len(self.instances)]
            elif self.routing_strategy == "round-robin":
                instance = self.instances[self.index % len(self.instances)]
                self.index += 1
            else:
                raise ValueError(f"Invalid routing strategy: {self.routing_strategy}")
            return instance

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        path = "/".join(filter(None, [self.prefix, key]))
        base_url = await self._get_instance(path)
        url = urljoin(f"{base_url}/", path)
        headers = {}

        if byte_range:
            raise NotImplementedError("Byte range requests not supported")

        async with self.session.get(url, headers=headers) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            data = await resp.read()
            return prototype.buffer.from_bytes(data)

    @property
    def supports_partial_reads(self) -> bool:
        return False

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        raise NotImplementedError("Partial reads not supported")

    async def exists(self, key: str) -> bool:
        base_url = await self._get_instance()
        path = "/".join(filter(None, [self.prefix, key]))
        url = urljoin(f"{base_url}/", path)
        async with self.session.head(url) as resp:
            return resp.status == 200

    @property
    def supports_writes(self) -> bool:
        return False

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("Read-only store")

    @property
    def supports_deletes(self) -> bool:
        return False

    async def delete(self, key: str) -> None:
        raise NotImplementedError("Read-only store")

    @property
    def supports_partial_writes(self) -> bool:
        return False

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, BytesLike]]
    ) -> None:
        raise NotImplementedError("Read-only store")

    @property
    def supports_listing(self) -> bool:
        return False

    def list(self) -> AsyncIterator[str]:
        raise NotImplementedError("Listing not supported")

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("Listing not supported")

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("Listing not supported")

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, ConsulLoadBalancedStore)
            and value.service_name == self.service_name
            and value.prefix == self.prefix
        )

    def close(self) -> None:
        super().close()
        if hasattr(self, "session"):
            asyncio.create_task(self.session.close())


async def test_main():
    store = ConsulLoadBalancedStore(
        consul_host="login33", service_name="xarray", prefix="datasets/inference/zarr"
    )

    resp = await store.get(".zmetadata", default_buffer_prototype())
    print(resp.to_bytes())


def open_zarr(
    consul_host: str,
    service_name: str,
    prefix: str,
    routing_strategy: Literal["hash", "round-robin"] = "hash",
    **kwargs,
):
    store = ConsulLoadBalancedStore(
        consul_host=consul_host,
        service_name=service_name,
        prefix=prefix,
        routing_strategy=routing_strategy,
    )
    return xarray.open_zarr(store, **kwargs)


def main():
    store = ConsulLoadBalancedStore(
        consul_host="login33", service_name="xarray", prefix="datasets/inference/zarr"
    )

    # resp = await store.get('.zmetadata', default_buffer_prototype())
    # print(resp.to_bytes())

    # Get metadata
    # g = zarr.open_group(store, mode="r")
    # print(g)
    # print(g['data'][0:])

    import xarray as xr

    ds = xr.open_zarr(store)
    mean = ds.isel(time=slice(0, 1000)).mean("time").compute()
    print(mean)

    # # Open group using sync wrapper
    # from zarr.core.sync import sync
    # group = zarr.api.asynchronous.open_group(store, mode="r")
    # group= zarr.api.synchronous.Group(sync(group))
    # print(group)


if __name__ == "__main__":
    asyncio.run(test_main())
    main()
