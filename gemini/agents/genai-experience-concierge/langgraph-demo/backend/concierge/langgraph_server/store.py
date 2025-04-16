# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
"""Utilities to load LangGraph stores from a supported config."""

import logging
from typing import Optional

from concierge.langgraph_server import schemas
from langgraph.store import base, memory
from langgraph.store.postgres import aio as postgres_aio
import psycopg
from psycopg.rows import DictRow, dict_row
import psycopg_pool

logger = logging.getLogger(__name__)


def load_store(
    backend_config: schemas.StoreConfig,
    index: Optional[base.IndexConfig] = None,
    ttl: Optional[base.TTLConfig] = None,
) -> base.BaseStore:
    """
    Loads a store based on the provided backend configuration.

    This function takes a StoreConfig object and returns a BaseStore instance,
    which can be a InMemoryStore, or AsyncPostgresStore, depending on the
    backend configuration.

    Args:
        backend_config: The configuration for the checkpoint backend.

    Returns:
        A BaseStore instance.

    Raises:
        ValueError: If an unknown backend type is provided.
    """

    store: base.BaseStore

    match backend_config:
        case schemas.MemoryBackendConfig():
            if ttl:
                raise ValueError("TTL is not supported for memory backend")

            # Create store
            store = memory.InMemoryStore(index=index)

        case schemas.PostgresBackendConfig():
            # Open postgres connection pool
            connection_pool = psycopg_pool.AsyncConnectionPool(
                conninfo=backend_config.dsn.unicode_string(),
                connection_class=psycopg.AsyncConnection[DictRow],
                kwargs={"autocommit": True, "row_factory": dict_row},
                open=False,
            )

            # Create store
            store = postgres_aio.AsyncPostgresStore(
                conn=connection_pool,
                index=index,
                ttl=ttl,
            )

        case _:
            raise ValueError(f"Unknown backend type: {type(backend_config)}")

    return store


async def setup_store(store: base.BaseStore) -> None:
    """
    Sets up the store, performing any necessary initialization.

    This function takes a BaseStore instance and performs any setup operations
    required for the specific type of store. For example, it opens the connection
    pool for Postgres.

    Args:
        store: The store to set up.
    """

    match store:
        case memory.InMemoryStore():
            pass

        case postgres_aio.AsyncPostgresStore():
            if isinstance(store.conn, psycopg_pool.AsyncConnectionPool):
                await store.conn.open()

            await store.setup()
            await store.start_ttl_sweeper()

        case _:
            logger.warning(f"Ignoring unknown checkpoint saver type: {type(store)}")


async def cleanup_store(store: base.BaseStore) -> None:
    """
    Cleans up the store, releasing any resources.

    This function takes a BaseStore instance and performs any cleanup operations
    required for the specific type of store. For example, it closes the connection
    pool for Postgres.

    Args:
        store: The store to clean up.
    """

    match store:
        case memory.InMemoryStore():
            pass

        case postgres_aio.AsyncPostgresStore():
            await store.conn.close()

        case _:
            logger.warning(f"Ignoring unknown store type: {type(store)}")
