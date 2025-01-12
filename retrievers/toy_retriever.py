from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class ToyRetriever(BaseRetriever):
    """A toy retriever that contains the top k documents that contain the user query.

   This retriever only implements the sync method _get_relevant_documents.

   If the retriever were to involve file access or network access, it could benefit
   from a native async implementation of `_aget_relevant_documents`.

   As usual, with Runnables, there's a default async implementation that's provided
   that delegates to the sync implementation running on another thread.
   """

    documents: List[Document]
    """List of documents to retrieve from."""
    k: int
    """Number of top results to return"""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Sync implementations for retriever."""
        matching_documents = []
        for document in self.documents:
            if len(matching_documents) > self.k:
                return matching_documents

            if query.lower() in document.page_content.lower():
                matching_documents.append(document)
        return matching_documents
