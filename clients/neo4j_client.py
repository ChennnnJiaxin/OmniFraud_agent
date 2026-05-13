from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from infra.config import AppConfig, load_app_config


class Neo4jClient:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_app_config()

    def create_driver(self):
        return GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_username, self.config.neo4j_password),
            database=self.config.neo4j_database,
        )

    def verify_connectivity(self) -> None:
        with self.create_driver() as driver:
            driver.verify_connectivity()

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self.create_driver() as driver:
            with driver.session() as session:
                result = session.run(cypher, **params)
                return result.data()

    def query_single(self, cypher: str, **params: Any) -> dict[str, Any] | None:
        with self.create_driver() as driver:
            with driver.session() as session:
                record = session.run(cypher, **params).single()
                return dict(record) if record else None
