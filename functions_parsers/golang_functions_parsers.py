import os
import re
from langchain_core.documents import Document

from functions_parsers.lang_functions_parsers import LanguageFunctionsParser


class GoLanguageFunctionsParser(LanguageFunctionsParser):

    def get_function_reserved_word(self) -> str:
        return "func"

    def is_searchable_file_name(self, function: Document) -> bool:
        file_path = str(function.metadata['source'])
        return "test" not in file_path[file_path.rfind("/") + 1:].split(".")[0].lower()

    def is_function(self, function: Document) -> bool:
        return function.page_content.startswith("func")

    def is_supported_file_extensions(self, extension: str) -> bool:
        return extension in self.supported_files_extensions()

    def supported_files_extensions(self) -> list[str]:
        return [".go"]

    def get_comment_line_notation(self, line: str) -> str:
        return "//"

    def dir_name_for_3rd_party_packages(self) -> str:
        return "vendor"

    def is_exported_function(self, function: Document) -> bool:
        function_name = self.get_function_name(function)
        return re.search("[A-Z][a-z0-9-]*", function_name)

    def get_function_name(self, function: Document) -> str:
        index_of_function_opening = function.page_content.index("{")
        function_header = function.page_content[:index_of_function_opening]
        # function is a method of a type
        if function_header.startswith("func ("):
            index_of_first_right_bracket = function_header.index(")")
            skip_receiver_arg = function_header[index_of_first_right_bracket:]
            index_of_first_left_bracket = skip_receiver_arg.index("(")
            return skip_receiver_arg[:index_of_first_left_bracket].strip()
        # regular function not tied to a certain type
        else:
            index_of_first_left_bracket = function_header.index("(")
            func_with_name = function_header[:index_of_first_left_bracket]
            return func_with_name.split(" ")[1]

    def search_for_called_function(self, caller_function: Document, callee_function: str,
                                   callee_function_package: str, code_documents: list[Document]) -> bool:
        index_of_function_opening = caller_function.page_content.index("{")
        index_of_function_closing = caller_function.page_content.rfind("{")
        caller_function_body = str(
            caller_function.page_content[index_of_function_opening + 1: index_of_function_closing])
        re.search("", caller_function_body)
        regex = f"([a-zA-Z_][a-zA-Z]+.)*{callee_function}\("
        matching = re.search(regex, caller_function_body, re.MULTILINE)
        if matching and matching.group(0):
            return self.__check_identifier_if_exists(function=caller_function, identifier_function=matching.group(0),
                                                     callee_package=callee_function_package,
                                                     code_documents=code_documents)
        else:
            return False

    def __check_identifier_if_exists(self, function: Document, identifier_function: str, callee_package: str,
                                     code_documents: list[Document]) -> bool:
        parts = identifier_function.split(".")
        # If there is no qualifier identifier , then check whether callee function and caller
        # function are in the same package
        if len(parts) == 1:
            if callee_package in self.get_package_names(function):
                return True
        # There is a qualifier identifier.
        else:
            identifier = parts[0]
            ## verify that identifier resolves to the package name. if identifier is imported in same file, and if so , if it's the same as callee package name
            for doc in code_documents:
                if function.metadata.get('source') == doc.metadata.get('source'):
                    # maybe identifier is the package itself in the file
                    regex = f"package {identifier}"
                    code_content = doc.page_content
                    matching = re.search(regex, code_content, re.MULTILINE)
                    if matching and matching.group(0):
                        return True

                    first_import = code_content.find("import")
                    last_import = code_content.rfind("import")
                    # block of imports
                    if first_import == last_import:
                        block_of_import = code_content[first_import + 6:]
                        after_left_bracket = block_of_import.find("(")
                        before_right_bracket = block_of_import.find(")")
                        inside_block = block_of_import[after_left_bracket + 1: before_right_bracket]
                        position_of_identifier_in_import_block = inside_block.find(identifier)
                        if position_of_identifier_in_import_block != -1:
                            row_of_identifier_and_rest = inside_block[position_of_identifier_in_import_block:]
                            index_of_end_of_line = row_of_identifier_and_rest.find(os.linesep)
                            row_of_identifier = row_of_identifier_and_rest[:index_of_end_of_line]
                            package_name_of_alias = row_of_identifier.split(" ")[1]
                            if package_name_of_alias.strip().lower() == callee_package.strip().lower():
                                return True
                    # Checks if there is a dedicated import with alias and imported package name
                    else:
                        identifier_import_position = code_content.find(f"import {identifier}")
                        if identifier_import_position != -1:
                            start_of_package_name = code_content[identifier_import_position + len("import ")
                                                                 + len(identifier):]
                            index_of_end_of_line = start_of_package_name.find(os.linesep)
                            package_name_to_check = start_of_package_name[:index_of_end_of_line]
                            if package_name_to_check.strip().lower() == callee_package.strip().lower():
                                return True

            ## otherwise, the identifier defined in the caller function.
            else:
                # TODO check if identifier is defined in the same function and dig into structures and identifiers defined by variables
                pass

        return False

    def get_package_names(self, function: Document) -> list[str]:
        package_names = list()
        full_doc_path = str(function.metadata['source'])
        parts = full_doc_path.split("/")
        if parts[0].startswith(self.dir_name_for_3rd_party_packages()):
            package_names.append(f"{parts[1]}/{parts[2]}")
            package_names.append(f"{parts[1]}/{parts[2]}/{parts[3]}")
        return package_names

    def get_package_name(self, function: Document, package_name: str) -> str:
        package_names = self.get_package_names(function)
        for package in package_names:
            if package_name.lower() in package.lower() and package_name.lower() == package.lower():
                return package.lower()
        return None

    def is_root_package(self, function: Document) -> bool:
        return not function.metadata['source'].startswith(self.dir_name_for_3rd_party_packages)

    def is_comment_line(self, line: str) -> bool:
        return line.strip().startswith("//")
