import os
import pickle
from langchain_core.documents import Document
from data_models.input import SourceDocumentsInfo
from retrievers.chain_of_calls_retriever import ChainOfCallsRetriever
from utils.dep_tree import Ecosystem
from utils.documents_loader import DocumentEmbedding


def process_list(documents_list):
    simplified_codes = 0
    functions_methods = 0
    others = 0
    for document in documents_list:
        if document.metadata.get('content_type') == 'functions_classes':
            functions_methods += 1
        elif document.metadata.get('content_type') == 'simplified_code':
            simplified_codes += 1
        else:
            others += 1
    print(f"simplified_codes: {simplified_codes}, functions_methods: {functions_methods}, others: {others}")


def create_documents(repository_url: str,
                     repository_digest: str):
    documents = list()
    cache_path = os.environ.get("DOCUMENTS_CACHE_PATH", "/home/zgrinber/poc_cache")

    repo_url = repository_url
    repo_digest = repository_digest

    cached_documents_path = \
        (f"{cache_path}/"
         f"{repo_url.replace('//', '.').replace('/', '.').replace(':', '')}-"
         f"{repo_digest}")
    if os.path.isfile(cached_documents_path):
        with open(cached_documents_path, 'rb') as doc_file:
            documents = pickle.load(doc_file)  # deserialize using load()
    else:

        document_embedder = DocumentEmbedding(embedding=None,
                                              vdb_directory="/tmp/vdb",
                                              git_directory="/tmp")

        documents = document_embedder.collect_documents(
            source_info=SourceDocumentsInfo(type='code', git_repo=repo_url,
                                            ref=("%s" % repo_digest), include=["**/*.go"],
                                            exclude=["**/*test*.go"]))
        with open(cached_documents_path, 'wb') as doc_file:  # open a text file
            pickle.dump(documents, doc_file)
    return documents


tests= [("https://github.com/kuadrant/authorino", "f792cd138891dc1ead99fd089aa757fbca3aace9",
         "crypto/x509,ParsePKCS1PrivateKey"),
        ("https://github.com/zvigrinberg/router", "b49f382f59d6479af9ea26f067ee5cb4e1dd13d9",
         "github.com/beorn7/perks,NewTargeted"),
        ("https://github.com/openshift/oc-mirror", "b137a53a5360a41a70432ea2bfc98a6cee6f7a4a",
         "github.com/mholt/archiver,Unarchive"),
        ("https://github.com/openshift/metallb","3be9d86e5752c6974a2fea99d9373af4f2225e6b",
         "github.com/jpillora/backoff,ForAttempt")]


for num, test in enumerate(tests):
    print(f"Test number #{num + 1}")
    print(f"Test parameters: {test}")
    (git_repo, git_commit_digest, the_input) = test
    documents_list = create_documents(repository_url=git_repo,
                                      repository_digest=git_commit_digest)

    retriever = ChainOfCallsRetriever(documents=documents_list, ecosystem=Ecosystem.GO, package_name="",
                                      manifest_path=f"/tmp/{git_repo}")
    process_list(documents_list)

    # call_hierarchy_list = retriever.invoke("github.com/beorn7/perks,NewTargeted")
    call_hierarchy_list = retriever.invoke(the_input)
    print("")
    print(f"Retriever found path={retriever.found_path}")
    print(f"path size={len(call_hierarchy_list)}")
    print("")
    print("==============================================")
    print("Prints Hierarchy call functions path")
    print("==============================================")
    print("")
    retriever.print_call_hierarchy(call_hierarchy_list)
    print("==============================================")
    print("Path Contents Content:")
    print("==============================================")
    print("")
    for i, function_method in enumerate(reversed(call_hierarchy_list)):
        print(f"File {i + 1}: {function_method.metadata['source']}")
        print("-------------------------------------------")
        print(function_method.page_content)
    print(" ")