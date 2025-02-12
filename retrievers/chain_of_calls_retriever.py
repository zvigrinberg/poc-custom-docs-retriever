from pathlib import Path
from typing import List, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from functions_parsers.lang_functions_parsers import LanguageFunctionsParser
from functions_parsers.lang_functions_parsers_factory import get_language_function_parser
from utils.dep_tree import DependencyTree, Ecosystem, get_dependency_tree_builder, ROOT_LEVEL_SENTINEL


def get_extension_of_file(file_path: str):
    extension_start = file_path.index(".")
    return file_path[extension_start:]


def get_functions_for_package(package_name: str, documents: list[Document], language_parser: LanguageFunctionsParser,
                              sources_location_packages=True):
    # Retrieve documents of package only ( functions + code)
    if sources_location_packages:
        for document in documents:
            doc_extension = get_extension_of_file(document.metadata.get('source'))
            if (document.metadata.get('source').startsWith(language_parser.dir_name_for_3rd_party_packages()) and
                    document.metadata.get('content_type') == 'functions_classes' and
                    language_parser.is_function(document) and
                    language_parser.is_exported_function(document) and
                    language_parser.is_supported_file_extensions(doc_extension) and
                    document_belongs_to_package(language_parser, document, package_name)):
                yield document
    # Retrieve documents of application only ( functions + code)
    else:
        for document in documents:
            doc_extension = get_extension_of_file(document.metadata.get('source'))
            if (language_parser.is_root_package(document)
                    and language_parser.is_function(document)
                    and language_parser.is_supported_file_extensions(doc_extension)):
                yield document


def document_belongs_to_package(language_parser: LanguageFunctionsParser, document: Document,
                                package_name: str) -> bool:
    return (not language_parser.is_root_package() and
            language_parser.get_package_names(document) == package_name.lower())


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

    documents: List[Document]
    """List of documents comprising the path of call, if exists."""
    documents_of_full_sources: List[Document]
    language_parser: LanguageFunctionsParser
    dependency_tree: DependencyTree
    tree_dict: dict
    manifest_path: Path
    package_name: str
    found_path: bool
    k: int = 10
    """Number of top results to return"""

    def __init__(self, documents: List[Document], ecosystem: Ecosystem, manifest_path: Path, package_name: str,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.dependency_tree = get_dependency_tree_builder(ecosystem)
        self.language_parser = get_language_function_parser(ecosystem)
        self.tree_dict = self.dependency_tree.builder.build_tree(manifest_path=manifest_path)
        self.package_name = package_name
        self.documents = documents
        self.found_path = False
        self.documents_of_full_sources = [doc for doc in self.documents
                                          if doc.metadata.get('content_type') == 'simplified_code']

    def __find_caller_function(self, document_function: Document, function_package: str) -> Document:
        function_name = self.language_parser.get_function_name(document_function)
        package_names = self.language_parser.get_package_names(document_function)
        direct_parents = list()
        # gets list of all direct parents of function
        for package_name in package_names:
            list_of_packages = self.tree_dict.get(package_name)
            if list_of_packages is not None:
                direct_parents.extend(list_of_packages)
        # gets list of documents to search in only from parents of function' package.
        relevant_docs_to_search_in = list()
        for package in direct_parents:
            sources_location_packages = True
            if package == ROOT_LEVEL_SENTINEL:
                sources_location_packages = False
            for doc in get_functions_for_package(package_name=package, documents=self.documents,
                                                 language_parser=self.language_parser,
                                                 sources_location_packages=sources_location_packages):
                relevant_docs_to_search_in.append(doc)
        for doc in relevant_docs_to_search_in:
            function_is_being_called = self.language_parser.search_for_called_function(caller_function=doc,
                                                                                       callee_function=
                                                                                       self.language_parser.
                                                                                       get_function_name(
                                                                                           document_function),
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
        current_function = query
        matching_documents = []
        for package in self.tree_dict:
            if self.package_name in package.lower():
                self.package_name = package
                break

        target_function_doc = find_initial_function(query, package_name=self.package_name, documents=self.documents,
                                                    language_parser=self.language_parser)
        matching_documents.append(target_function_doc)
        end_loop = False
        current_package_name = self.package_name
        while True:
            if end_loop:
                break
            found_document = self.__find_caller_function(doc=target_function_doc, function_package=current_package_name)
            if found_document is not None:
                matching_documents.append(found_document)
                if self.language_parser.is_root_package(found_document):
                    end_loop = True
                    self.found_path = True
                else:
                    target_function_doc = found_document
                    package_names = [package_name for package_name in self.language_parser.get_package_names()
                                     if self.tree_dict.get(package_name, None) is not None]
                    current_package_name = package_names[0]
            else:
                end_loop = True

        return matching_documents
