import os
import pickle
import re

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


def handle_argument(param: str) -> str:
    if (param.startswith("'") and param.endswith("'")) or (param.startswith('"') and param.endswith('"')):
        return param
    elif param.__contains__("(") and param.__contains__(")"):
        function_prefix_index_end = param.find("(") + 1
        function_ending_index_end = param.rfind(")")
        return traverse_all_parameters(function_ending_index_end, function_prefix_index_end, param)
    else:
        return ".*"


def infer_if_short_package_name_match(package: str, package_names: list[str], final_package: set[str]) -> bool:
    package_with_version_suffix = r"[./][vV][1-9]+$"
    match = re.search(package_with_version_suffix, package)
    if match:
        parts = package.split(match.group())
        short_package_name = parts[-2].split("/")[-1]
    else:
        parts = package.split("/")
        short_package_name = parts[-1].split("/")[-1]

    for current_package in package_names:
        if short_package_name in current_package:
            final_package.add(current_package)
            return True

    else:
        return False


def extract_using_function_name(input_string: str):
    parts_input = input_string.split(",", 1)
    package = parts_input[0]
    function = parts_input[1]

    # call_hierarchy_list = retriever.invoke("github.com/beorn7/perks,NewTargeted")
    final_package = set()
    package_functions = [doc for doc in documents if
                         lang_parser.get_package_name(doc, package)
                         or
                         infer_if_short_package_name_match(package, lang_parser.get_package_names(doc), final_package)]

    the_final_package = list(final_package)
    if len(the_final_package) > 0 and the_final_package[0] != package:
        package = the_final_package[0]

    if function.__contains__(".") and "(" or ")" in function:
        function_string = str(function)
        function_prefix_index_end = function_string.find("(") + 1
        function_ending_index_end = function_string.rfind(")")
        function_builder = traverse_all_parameters(function_ending_index_end, function_prefix_index_end,
                                                   function_string)
    else:
        return input_string
    new_function_name = ""
    comment_line_character = lang_parser.get_comment_line_notation()
    for func in package_functions:

        if match := re.search(rf"{function_builder}", func.page_content, flags=re.MULTILINE):
            current_offset = match.start() - 1
            while func.page_content[current_offset] != os.linesep:
                current_offset -= 1
            if func.page_content[current_offset: match.start()].strip().startswith(comment_line_character):
                pass
            else:
                new_function_name = lang_parser.get_function_name(func)
                break
    new_input = f"{package},{new_function_name}"
    return new_input


def traverse_all_parameters(function_ending_index_end, function_prefix_index_end, function_string):
    current_idx = function_prefix_index_end
    function_builder = function_string[:function_prefix_index_end]
    if current_idx == function_ending_index_end:
        function_builder += ")"
    while current_idx < function_ending_index_end:
        end_of_arg_ind = function_string[current_idx:].find(",")
        if end_of_arg_ind > -1:
            value = handle_argument(function_string[current_idx: current_idx + end_of_arg_ind].strip())
            function_builder += f"\\s?{value},"
            current_idx = current_idx + end_of_arg_ind + 1
        else:
            # last argument
            value = handle_argument(function_string[current_idx:function_ending_index_end].strip())
            function_builder += f"\\s?{value}\\s?)"
            current_idx = function_ending_index_end

    return function_builder


# ("https://github.com/openshift/oauth-server", "c055dbb9a84e04575ade106e9a43cc638a8aeaef",
#  'github.com/go-jose/go-jose/v4,strings.Split(token, ".")'),
# ("https://github.com/kuadrant/authorino", "f792cd138891dc1ead99fd089aa757fbca3aace9",
#  "crypto/rsa,Verify"),
tests = [
    ("https://github.com/openshift/oauth-server", "c055dbb9a84e04575ade106e9a43cc638a8aeaef",
     'github.com/go-jose/go-jose/v4,strings.Split(token, ".")'),
         ("https://github.com/openshift/assisted-installer", "bc16edd293be0a684ae0a97fd9dc27d0ebe8fd90",
          'github.com/jackc/pgx/v4,pgx.Connect(context.Background(), os.Getenv("DATABASE_URL"))'),
         ("https://github.com/kuadrant/authorino", "f792cd138891dc1ead99fd089aa757fbca3aace9",
          "crypto/x509,ParsePKCS1PrivateKey"),

         ("https://github.com/zvigrinberg/router", "b49f382f59d6479af9ea26f067ee5cb4e1dd13d9",
          "github.com/beorn7/perks,NewTargeted"),
         ("https://github.com/openshift/oc-mirror", "b137a53a5360a41a70432ea2bfc98a6cee6f7a4a",
          "github.com/mholt/archiver,Unarchive"),
         ("https://github.com/openshift/metallb", "3be9d86e5752c6974a2fea99d9373af4f2225e6b",
          "github.com/jpillora/backoff,ForAttempt")]

for num, test in enumerate(tests):
    print(f"Test number #{num + 1}")
    print(f"Test parameters: {test}")
    (git_repo, git_commit_digest, the_input) = test
    documents_list = create_documents(repository_url=git_repo,
                                      repository_digest=git_commit_digest)

    retriever = ChainOfCallsRetriever(documents=documents_list, ecosystem=Ecosystem.GO, package_name="",
                                      manifest_path=f"/tmp/{git_repo}")
    lang_parser = retriever.language_parser
    documents = retriever.documents

    process_list(documents_list)
    the_input = extract_using_function_name(the_input)

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
