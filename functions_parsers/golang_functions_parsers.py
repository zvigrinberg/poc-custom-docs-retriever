import re
from langchain_core.documents import Document

from functions_parsers.lang_functions_parsers import LanguageFunctionsParser


class GoLanguageFunctionsParser(LanguageFunctionsParser):

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

    def search_for_called_function(self, caller_function: Document, callee_function: str) -> bool:
        pass

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

    def is_root_package(self, function: Document) -> bool:
        return not function.metadata['source'].startswith(self.dir_name_for_3rd_party_packages)

    def is_comment_line(self, line: str) -> bool:
        return line.strip().startswith("//")
