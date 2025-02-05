from langchain_core.documents import Document

from data_models.input import SourceDocumentsInfo
from retrievers.toy_retriever import ToyRetriever
from utils.documents_loader import DocumentEmbedding

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"type": "dog", "trait": "loyalty"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"type": "cat", "trait": "independence"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"type": "fish", "trait": "low maintenance"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"type": "bird", "trait": "intelligence"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"type": "rabbit", "trait": "social"},
    ),
]


def process_list(documents_list):
    for document in documents_list:
        print(document)


retriever = ToyRetriever(documents=documents, k=3)
print(retriever.invoke("that"))

document_embedder = DocumentEmbedding(embedding=None,
                                      vdb_directory="/tmp/vdb",
                                      git_directory="/tmp")
documents_list = document_embedder.collect_documents(
    source_info=SourceDocumentsInfo(type='code', git_repo="https://github.com/zvigrinberg/router",
                                    ref="b49f382f59d6479af9ea26f067ee5cb4e1dd13d9", include=["**/*.go"]))

process_list(documents_list)
