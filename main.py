from langchain_core.documents import Document

from data_models.input import SourceDocumentsInfo
from retrievers.chain_of_calls_retriever import ChainOfCallsRetriever
from utils.dep_tree import Ecosystem
from utils.documents_loader import DocumentEmbedding

def process_list(documents_list):
    simplified_codes = 0
    functions_parsers = 0
    others = 0
    for document in documents_list:
        if document.metadata.get('content_type') == 'functions_classes':
            functions_parsers += 1
        elif document.metadata.get('content_type') == 'simplified_code':
            simplified_codes += 1
        else:
            others += 1
    print(f"simplified_codes: {simplified_codes}, functions_parsers: {functions_parsers}, others: {others}")


document_embedder = DocumentEmbedding(embedding=None,
                                      vdb_directory="/tmp/vdb",
                                      git_directory="/tmp")
documents_list = document_embedder.collect_documents(
    source_info=SourceDocumentsInfo(type='code', git_repo="https://github.com/zvigrinberg/router",
                                    ref="b49f382f59d6479af9ea26f067ee5cb4e1dd13d9", include=["**/*.go"],
                                    exclude=["**/*test*.go"]))

retriever = ChainOfCallsRetriever(documents=documents_list, ecosystem=Ecosystem.GO, package_name="",
                                  manifest_path="/tmp/https:/github.com/zvigrinberg/router")
process_list(documents_list)

print(retriever.invoke("github.com/beorn7/perks,NewTargeted"))


