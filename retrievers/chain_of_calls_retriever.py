import re
from pathlib import Path
from typing import List, Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from functions_parsers.lang_functions_parsers import LanguageFunctionsParser
from functions_parsers.lang_functions_parsers_factory import get_language_function_parser
from utils.dep_tree import DependencyTree, Ecosystem, get_dependency_tree_builder, ROOT_LEVEL_SENTINEL


def get_extension_of_file(file_path: str):
    extension_start = file_path.rfind(".")
    return file_path[extension_start:]


def get_functions_for_package(package_name: str, documents: list[Document], language_parser: LanguageFunctionsParser,
                              function_to_search="", sources_location_packages=True, ):
    # Retrieve documents of packages only ( functions + code)
    if sources_location_packages:
        for document in documents:
            doc_extension = get_extension_of_file(document.metadata.get('source'))
            if (document.metadata.get('source').startswith(language_parser.dir_name_for_3rd_party_packages()) and
                    document.metadata.get('content_type') == 'functions_classes' and
                    language_parser.is_function(document) and
                    language_parser.is_exported_function(document) and
                    language_parser.is_supported_file_extensions(doc_extension) and
                    document_belongs_to_package(language_parser, document, package_name) and
                    function_called_from_caller_body(document, function_to_search, language_parser)):
                yield document
    # Retrieve documents of application only ( functions + code)
    else:
        for document in documents:
            doc_extension = get_extension_of_file(document.metadata.get('source'))
            if (language_parser.is_root_package(document)
                    and document.metadata.get('content_type') == 'functions_classes'
                    and language_parser.is_function(document)
                    and language_parser.is_supported_file_extensions(doc_extension)
                    and function_called_from_caller_body(document, function_to_search, language_parser)):
                yield document


def function_called_from_caller_body(document, function_to_search, language_parser: LanguageFunctionsParser) -> bool:
    if function_to_search.strip() == "":
        return True
    function_word = language_parser.get_function_reserved_word()
    func_header_template = rf"${function_word} (\(.*\))?\s?${function_to_search}"
    return (re.search(pattern=function_to_search, string=document.page_content,
                      flags=re.IGNORECASE | re.MULTILINE)
            # verify caller function or method is not the function
            and not re.search(pattern=func_header_template,
                              string=document.page_content,
                              flags=re.IGNORECASE | re.MULTILINE))


def document_belongs_to_package(language_parser: LanguageFunctionsParser, document: Document,
                                package_name: str) -> bool:
    return (not language_parser.is_root_package(document) and
            language_parser.get_package_name(function=document, package_name=package_name))


def find_initial_function(function_name: str, package_name: str, documents: list[Document],
                          language_parser: LanguageFunctionsParser) -> Document:
    for index, document in enumerate(get_functions_for_package(package_name, documents, language_parser)):
        # document_function_calls_input_function = True
        if function_name.lower() == language_parser.get_function_name(document).lower():
            # if language_parser.search_for_called_function(document, callee_function=function_name):
            return document


class ChainOfCallsRetriever(BaseRetriever):
    """A ChainOfCall retriever that contains the top k documents that contain the user query.

   This retriever only implements the sync method _get_relevant_documents.

   If the retriever were to involve file access or network access, it could benefit
   from a native async implementation of `_aget_relevant_documents`.

   As usual, with Runnables, there's a default async implementation that's provided
   that delegates to the sync implementation running on another thread.
   """

    documents: List[Document] | None
    """List of documents comprising the path of call, if exists."""
    documents_of_full_sources: List[Document] | None
    language_parser: Optional[LanguageFunctionsParser]
    dependency_tree: Optional[DependencyTree]
    tree_dict: Optional[dict]
    ecosystem: Optional[Ecosystem]
    manifest_path: Optional[Path]
    package_name: str
    found_path: Optional[bool]
    k: int = 10
    """Number of top results to return"""

    def __init__(self, documents: List[Document], ecosystem: Ecosystem, manifest_path: Path,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.ecosystem = ecosystem
        self.dependency_tree = DependencyTree(ecosystem=ecosystem)
        self.language_parser = get_language_function_parser(ecosystem)
        self.manifest_path = manifest_path
        if self.dependency_tree.builder is None:
            raise RuntimeError("Couldn't continue as dependencies wasn't generated")

        self.tree_dict = self.dependency_tree.builder.build_tree(manifest_path=manifest_path)
        self.documents = documents
        self.found_path = False
        self.documents_of_full_sources = [doc for doc in self.documents
                                          if doc.metadata.get('content_type') == 'simplified_code']

    def __find_caller_function(self, document_function: Document, function_package: str) -> Document:
        package_names = self.language_parser.get_package_names(document_function)
        direct_parents = list()
        # gets list of all direct parents of function
        for package_name in package_names:
            list_of_packages = self.tree_dict.get(package_name)
            if list_of_packages is not None:
                direct_parents.extend(list_of_packages)
            # Add same package itself to search path.
        direct_parents.extend([function_package])
        # gets list of documents to search in only from parents of function' package.
        relevant_docs_to_search_in = list()
        for package in direct_parents:
            sources_location_packages = True
            if package == ROOT_LEVEL_SENTINEL:
                sources_location_packages = False
            function_name_to_search = self.language_parser.get_function_name(document_function)
            for doc in get_functions_for_package(package_name=package, documents=self.documents,
                                                 language_parser=self.language_parser,
                                                 sources_location_packages=sources_location_packages,
                                                 function_to_search=function_name_to_search):
                relevant_docs_to_search_in.append(doc)
        for doc in relevant_docs_to_search_in:
            function_is_being_called = self.language_parser.search_for_called_function(caller_function=doc,
                                                                                       callee_function=
                                                                                       function_name_to_search,
                                                                                       callee_function_package=
                                                                                       function_package,
                                                                                       code_documents=
                                                                                       self.documents_of_full_sources
                                                                                       )
            if function_is_being_called:
                return doc

        return None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Sync implementations for retriever."""
        (package_name, function) = tuple(query.split(","))
        matching_documents = []
        for package in self.tree_dict:
            if package_name in package.lower():
                package_name = package
                break

        target_function_doc = find_initial_function(function, package_name=package_name, documents=self.documents,
                                                    language_parser=self.language_parser)
        matching_documents.append(target_function_doc)
        end_loop = False
        current_package_name = package_name
        while True:
            if end_loop:
                break
            found_document = self.__find_caller_function(document_function=target_function_doc,
                                                         function_package=current_package_name)
            if found_document is not None:
                matching_documents.append(found_document)
                if self.language_parser.is_root_package(found_document):
                    end_loop = True
                    self.found_path = True
                else:
                    target_function_doc = found_document
                    package_names = [package_name for package_name in
                                     self.language_parser.get_package_names(target_function_doc)
                                     if self.tree_dict.get(package_name, None) is not None]
                    current_package_name = package_names[0]
            else:
                end_loop = True

        return matching_documents

    def print_call_hierarchy(self, call_hierarchy_list: list[Document]):
        for i , package_function in enumerate(reversed(call_hierarchy_list)):
            packages_names = self.language_parser.get_package_names(package_function)
            maximum_length_package = max(len(packages_names[0]), len(packages_names[1]))
            if maximum_length_package == len(packages_names[0]):
                package_name = packages_names[0]
            else:
                package_name = packages_names[1]
            function_name = self.language_parser.get_function_name(package_function)
            print(f"(package={package_name},function={function_name},depth={i})")
