from copy import deepcopy
from streamlit.connections import ExperimentalBaseConnection
from streamlit.connections.util import extract_from_dict
from streamlit.errors import StreamlitAPIException
from qdrant_client import QdrantClient
from collections import ChainMap
from typing import List, Any

COLLECTION_NAME = "beans"

_ALL_CONNECTION_PARAMS = [
                    "url",
                    "api_key",
                    ]

class QdrantConnection(ExperimentalBaseConnection[QdrantClient]):

    def _connect(self, **kwargs) -> QdrantClient:

        kwargs = deepcopy(kwargs)
        conn_param_kwargs = extract_from_dict(_ALL_CONNECTION_PARAMS, kwargs)
        conn_params = ChainMap(conn_param_kwargs, self._secrets.to_dict())
        
        if not len(conn_params):
            raise StreamlitAPIException(
                "Missing qdrant DB connection configuration. "
                "Did you forget to set this in `secrets.toml` or as kwargs to `st.experimental_connection`?"
            )
        client = QdrantClient(
                url=conn_params["url"],
                api_key=conn_params["api_key"],
        )
        return client
    

    @property
    def client(self) -> QdrantClient:
        """Access the underlying AbstractFileSystem for full API operations."""
        return self._instance
   

    def find_similars(self, query_vector: List[float], limit: int = 5) -> List[dict[str, Any]]:
        hits = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )

        results = [{'id':result.id, 
            "path": result.payload["path"],
            "score": result.score, 
            "label": result.payload["label"]
            } 
            for result in hits]

        return results
