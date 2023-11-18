"""
Arize-Phoenix LlamaPack.
"""

from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List

from llama_index import set_global_handler
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.llama_pack.base import BaseLlamaPack
from llama_index.schema import TextNode

if TYPE_CHECKING:
    from phoenix import Session as PhoenixSession


class ArizePhoenixQueryEnginePack(BaseLlamaPack):
    """
    The Arize-Phoenix LlamaPack show how to instrument your LlamaIndex query
    engine for tracing and observability for your LLM application. It launches
    Phoenix in the background, builds an index over an input list of nodes, and
    instantiates and instruments a query engine over that index so that trace
    data from each query is sent to Phoenix.

    Note: Using this LlamaPack requires that your OpenAI API key is set via the
    OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        nodes: List[TextNode],
        **kwargs: Any,
    ) -> None:
        """
        Initializes a new instance of ArizePhoenixQueryEnginePack.

        Args:
            nodes (List[TextNode]): An input list of nodes over which the index
            will be built.
        """
        set_global_handler("arize_phoenix")
        phoenix = _import_phoenix()
        self._session: PhoenixSession = phoenix.launch_app()
        self._index = VectorStoreIndex(nodes, **kwargs)
        self._query_engine = self._index.as_query_engine()

    def get_modules(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the internals of the LlamaPack.

        Returns:
            Dict[str, Any]: A dictionary containing the internals of the
            LlamaPack.
        """
        return {
            "session": self._session,
            "session_url": self._session.url,
            "index": self._index,
            "query_engine": self._query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs queries against the index.

        Returns:
            Any: A response from the query engine.
        """
        return self._query_engine.query(*args, **kwargs)


def _import_phoenix() -> ModuleType:
    """
    Imports the arize-phoenix module or raises an ImportError if it cannot be
    found.

    Raises:
        ImportError: If the arize-phoenix package could not be found.

    Returns:
        ModuleType: The arize-phoenix module.
    """
    try:
        import phoenix

        return phoenix
    except ImportError:
        raise ImportError(
            "The arize-phoenix package could not be found. "
            "Please install with `pip install arize-phoenix`."
        )
