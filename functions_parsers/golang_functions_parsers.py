import os
import re
from langchain_core.documents import Document

from functions_parsers.lang_functions_parsers import LanguageFunctionsParser

EMBEDDED_TYPE = "embedded_type"

PARAMETER = "parameter"

RETURN_TYPES = "return_types"

LOCAL_VAR_USAGE = "local_var_usage"
LOCAL_IMPLICIT = "local_implicit"
LOCAL_INDIRECT_TYPES_INDICATIONS = [LOCAL_VAR_USAGE, LOCAL_IMPLICIT]
PRIMITIVE_TYPES = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64',
                   'complex64', 'complex128', 'byte', 'rune', 'uint', 'int', 'uintptr', 'string', 'ptr', 'bool']
MAP_TYPE_REGEX = r"map\[[a-zA-Z0-9]+\][a-zA-Z0-9]+ "
SLICE_TYPES_REGEX = r"\[\][a-zA-Z0-9]+ "


# def parse_type_struct(fieldName: str,  ) -> : dict[str,str]

def get_package_name_file(function: Document):
    content = function.page_content
    index_of_package = content.find("package")
    index_of_end_of_line = index_of_package + content[index_of_package:].find(os.linesep)
    if index_of_package > -1:
        return content[index_of_package + 8:index_of_end_of_line].strip()


def check_types_from_callee_package(params: list[tuple], type_documents: list[Document], callee_package: str,
                                    code_documents: dict[str, Document], parameter: str,
                                    callee_function_file_name: str) -> bool:
    callee_function_doc = code_documents.get(callee_function_file_name)
    callee_function_file_package_name = get_package_name_file(callee_function_doc)
    for param in params:
        param_name, param_type = param
        if param_name == parameter:
            param_type_stripped = str(param_type).replace("*", "").replace("&", "")
            parts = param_type_stripped.split(".")
            # Only type without package
            if len(parts) == 1:
                for the_type in type_documents:
                    if the_type.page_content.startwith(f"type {parts[0]}"):
                        code_with_type_file = code_documents.get(the_type.metadata['source'])
                        type_file_package_name = get_package_name_file(code_with_type_file)
                        if type_file_package_name == callee_function_file_package_name:
                            return True
            # type with package qualifier
            else:
                for the_type in type_documents:
                    if the_type.page_content.startwith(f"type {parts[1]}"):
                        code_with_type_file = code_documents.get(the_type.metadata['source'])
                        package_match = handle_imports(code_with_type_file, parts[0], callee_package)
                        return package_match
                else:
                    return False


def handle_imports(code_content: str, identifier: str, callee_package: str) -> bool:
    start_search_for_import = max(code_content.rfind("//import"), code_content.rfind("// import"))
    if start_search_for_import > -1:
        after_last_import_comment_index = start_search_for_import + 4
    else:
        after_last_import_comment_index = 0
    after_last_import_comment = code_content[after_last_import_comment_index:]
    first_import = after_last_import_comment.find("import")
    last_import = after_last_import_comment.rfind("import")
    # return after_last_import_comment, first_import, last_import
    if first_import == last_import:
        block_of_import = after_last_import_comment[first_import + 6:]
        after_left_bracket = block_of_import.find("(")
        before_right_bracket = block_of_import.find(")")
        inside_block = block_of_import[after_left_bracket + 1: before_right_bracket]
        position_of_identifier_in_import_block = inside_block.find(identifier)
        if position_of_identifier_in_import_block != -1:
            previous_end_of_line = inside_block[:position_of_identifier_in_import_block].rfind(
                os.linesep)
            alias_package_extended = inside_block[previous_end_of_line:].strip()
            index_of_end_of_line = alias_package_extended.find(os.linesep)
            row_of_identifier = alias_package_extended[:index_of_end_of_line]
            if len(row_of_identifier.split(" ")) > 1:
                package_name_of_alias = row_of_identifier.split(" ")[1].strip('"')
            else:
                package_name_of_alias = row_of_identifier.split(" ")[0].strip('"')

            if (package_name_of_alias.strip().lower() == callee_package.strip().lower()
                    or callee_package.strip().lower() in package_name_of_alias.strip().lower()):
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
        #  import without alias, in this case maybe package name contain alias
        else:
            # re.search(regex, caller_function_body, re.MULTILINE)
            matching = re.search(rf"import [\'\"].*{identifier}[\'\"]", code_content)
            if matching and matching.group(0):
                import_line = code_content[matching.start():]
                import_package_line = import_line[:import_line.find(os.linesep)].strip()
                package_name = import_package_line.split(r"\s")[1]
                if package_name.strip().lower() == callee_package.strip().lower():
                    return True
    return False


class GoLanguageFunctionsParser(LanguageFunctionsParser):

    def __trace_down_package(self, expression: str, code_documents: dict[str, Document], type_documents: list[Document],
                             callee_package: str, fields_of_types: dict[tuple, list[tuple]],
                             functions_local_variables_index: dict[str, dict],
                             caller_function_index: str) -> bool:
        variables_mappings = functions_local_variables_index[caller_function_index]
        parts = expression.split(".")
        result = False

        (resolved_type, struct_initializer_expression,
         value, var_properties) = self.__prepare_package_lookup(parts, variables_mappings, the_part=-1)

        if (var_properties is not None
                and (struct_initializer_expression or
                     resolved_type not in LOCAL_INDIRECT_TYPES_INDICATIONS or value == PARAMETER)):
            result = self.__lookup_package(callee_package, resolved_type, struct_initializer_expression,
                                           type_documents, value)

        # Property/member is not in function, check if it's member/property of a type
        elif var_properties is None and len(parts) > 1:
            (resolved_type, struct_initializer_expression,
             value, var_properties) = self.__prepare_package_lookup(parts, variables_mappings, the_part=-2)
            if (var_properties is not None
                    and (resolved_type not in LOCAL_INDIRECT_TYPES_INDICATIONS or value == PARAMETER)):
                field_name = parts[-1]
                possible_types = {key: value for (key, value) in fields_of_types.items()
                                  if resolved_type in key
                                  and any([field for field in value if field_name in field])}
                for the_type, mappings in possible_types.items():
                    for mapping in mappings:
                        if field_name in mapping:
                            returned_matched_types = self.__get_type_docs_matched_with_callee_package(callee_package,
                                                                                                      mapping[1],
                                                                                                      type_documents)
                            if len(returned_matched_types) > 0:
                                result = True

        elif var_properties is not None:
            value = var_properties.get("value", None)
            if value is not None and value.strip() != "":
                return self.__trace_down_package(value, code_documents, type_documents, callee_package,
                                                 fields_of_types, functions_local_variables_index,
                                                 caller_function_index)

        return result

    def __prepare_package_lookup(self, parts, variables_mappings, the_part: int):
        var_properties = variables_mappings.get(parts[the_part], None)
        if var_properties is not None:
            resolved_type = var_properties.get("type")
            value = var_properties.get("value")
            struct_initializer_expression = re.search(r"(&|\\*)?\w+\s*{", value)
            resolved_type = str(resolved_type).replace("&", "").replace("*", "")
            return resolved_type, struct_initializer_expression, value, var_properties
        else:
            return None, None, None, None

    def __lookup_package(self, callee_package, resolved_type, struct_initializer_expression, type_documents,
                         value) -> bool:
        result = False
        if not struct_initializer_expression and resolved_type not in PRIMITIVE_TYPES:
            docs = self.__get_type_docs_matched_with_callee_package(callee_package, resolved_type, type_documents)

            if len(docs) > 0:
                result = True
            elif value == PARAMETER:
                result = False

        elif struct_initializer_expression:
            struct_type = (struct_initializer_expression.group(0).replace("{", "")
                           .replace("&", "").replace("*", ""))
            docs = self.__get_type_docs_matched_with_callee_package(callee_package, struct_type, type_documents)
            if len(docs) > 0:
                result = True
        return result

    def __get_type_docs_matched_with_callee_package(self, callee_package, checked_type, type_documents) -> list[
        Document]:
        return [a_type for a_type in type_documents if callee_package in a_type.metadata['source'] and
                (self.__get_type_name(a_type) == checked_type or self.__get_type_name(a_type) in checked_type)]

    def create_map_of_local_vars(self, functions_methods_documents: list[Document]) -> dict[str, dict]:
        mappings = dict()
        for func_method in functions_methods_documents:
            func_key = f"{self.get_function_name(func_method)}@{func_method.metadata['source']}"
            all_vars = dict()
            for row in func_method.page_content.splitlines():
                if not self.is_comment_line(row):
                    # Extract arguments and receiver argument of type as parameters
                    if row.startswith("func"):
                        match = re.finditer(r"(func|\w+)\s*\([a-zA-Z0-9\s*,.\[\]]+\)"
                                            , func_method.page_content[:func_method.page_content.find("{")]
                                            , flags=re.MULTILINE)
                        for current_match in match:
                            current_args = (current_match.group(0).replace("\n\t", "")
                                            .replace("\t", "").replace("\n", ""))
                            current_params = re.search(r"\(.*\)", current_args)
                            params = (current_params.group(0).replace("(", "")
                                      .replace(")", "").split(","))
                            params_tuple = tuple(params)
                            for param in reversed(params_tuple):
                                data = param.strip().split(" ")
                                param_name = data[0]
                                if len(data) > 1:
                                    param_type = data[1]
                                the_value = PARAMETER
                                all_vars[param_name] = {"value": the_value, "type": param_type}
                        # Gets return types from function
                        index_of_start_func = func_method.page_content.find(row)
                        last_left_curly_bracket = index_of_start_func + func_method.page_content.find("{")
                        last_right_bracket = func_method.page_content[
                                             index_of_start_func:last_left_curly_bracket].rfind(")")
                        return_parameters = func_method.page_content[index_of_start_func + last_right_bracket + 1:
                                                                     last_left_curly_bracket - 1].strip()
                        return_parameters = return_parameters.replace(")", "").replace("(", "")
                        if return_parameters.strip() != "":
                            all_vars[RETURN_TYPES] = return_parameters.split(",")
                        else:
                            all_vars[RETURN_TYPES] = []


                    elif row.strip().startswith("var ") and not re.search(r"var\s*\(", row.strip()):
                        row_without_var_prefix = row.strip()[3:].strip()
                        parts = row_without_var_prefix.split()
                        # variable name
                        left_side = parts[0]
                        if len(parts) == 2 and parts[1].__contains__("="):
                            assignment = row.strip().split("=")
                            all_vars[(left_side.strip())] = {"value": assignment[1],
                                                             "type": assignment[0].replace("*", "")}
                        else:
                            if len(parts) == 2:
                                all_vars[(left_side.strip())] = {"value": "", "type": parts[1].replace("*", "")}
                    elif row.strip().__contains__(":=") and row.strip().__contains__("if"):
                        index_of_start_if = func_method.page_content.find(row)
                        # Go until delimiter of assignment and start of boolean expression
                        end_of_assignment = func_method.page_content[index_of_start_if:].find(";")
                        parts = row.strip().split(":=")
                        left_side = parts[0].replace("if", "")
                        right_side = func_method.page_content[index_of_start_if + 2:
                                                              index_of_start_if + end_of_assignment - 1].strip()

                        all_vars[(left_side.strip())] = {" value": right_side.strip().replace
                        ("\n\t", "").replace("\t", ""), "type": LOCAL_IMPLICIT
                                                         }

                    elif (row.strip().__contains__(":=") and not row.strip().__contains__("if")
                          and not row.strip().__contains__("for ") and not row.strip().__contains__("range ")
                          and not row.strip().__contains__("range ") and not row.strip().__contains__("case ")):
                        parts = row.strip().split(":=")
                        left_side = parts[0]
                        right_side = parts[1]
                        all_vars[(left_side.strip())] = {"value": right_side.strip(), "type": LOCAL_IMPLICIT}
                    elif (row.strip().__contains__("=") and not row.strip().__contains__("==")
                          and not row.strip().__contains__("!=")
                          and not row.strip().__contains__("if") and not row.strip().__contains__("for ")
                          and not row.strip().__contains__("range ") and not row.strip().__contains__("case ")):
                        parts = row.strip().split("=")
                        left_side = parts[0]
                        right_side = parts[1]
                        all_vars[(left_side.strip())] = {"value": right_side.strip(), "type": ("%s" % LOCAL_VAR_USAGE)}
                    else:
                        pass

            mappings[func_key] = all_vars

        return mappings

    def __is_struct_or_interface_type(self, doc: Document):
        return re.search(r"^type\s+[a-zA-Z0-9_]+\s+(struct|interface)", doc.page_content)

    def __get_type_kind(self, the_type: Document) -> str:
        if self.__is_struct_or_interface_type(the_type):
            parts_of_type_header = the_type.page_content.split()
            return parts_of_type_header[2] if len(parts_of_type_header) > 2 else ""
        #  not composite type, either primitive or wrapper
        else:
            if match := re.search(r"^type\s+ \(", the_type.page_content):
                right_bracket = match.group(0).find(")")
                left_bracket = match.group(0).find("(")
                parts = match.group(0)[left_bracket:right_bracket].strip().split(sep=" ", maxsplit=1)
                return parts[1]
            else:
                parts = the_type.page_content.split(sep=" ", maxsplit=2)  # if the_type.page_content
                return parts[2]

    def __get_type_name(self, the_type: Document) -> str:
        if self.__is_struct_or_interface_type(the_type):
            parts_of_type_header = the_type.page_content.split()
            return parts_of_type_header[1] if len(parts_of_type_header) > 1 else ""
        elif match := re.search(r"^type\s+ \(", the_type.page_content):
            right_bracket = match.group(0).find(")")
            left_bracket = match.group(0).find("(")
            parts = match.group(0)[left_bracket:right_bracket].strip().split(sep=" ", maxsplit=1)
            return parts[0]
        else:
            parts = the_type.page_content.split(sep=" ", maxsplit=2)  # if the_type.page_content
            return parts[1]

    def parse_all_type_struct_class_to_fields(self, types: list[Document]) -> dict[tuple, list[tuple]]:
        types_mapping = dict()
        for the_type in types:
            the_kind = self.__get_type_kind(the_type)
            type_name = self.__get_type_name(the_type)
            type_key = type_name
            if the_kind.lower() == "interface":
                type_key = f"interface;{type_name}"

            start_index = the_type.page_content.find("{") + 2
            end_index = the_type.page_content.rfind("}")
            current_index = start_index
            fields_list = list()
            if end_index > - 1:
                while current_index < end_index:
                    index_of_eol = the_type.page_content[current_index:].find(os.linesep)
                    end_of_row = end_index
                    if index_of_eol > -1:
                        end_of_row = index_of_eol
                    current_line = the_type.page_content[current_index:current_index + end_of_row]
                    if not self.is_comment_line(current_line) and not current_line.strip() == "":
                        right_bracket_completion = ""
                        if type_key.startswith("interface"):
                            parts = current_line.split(sep=")", maxsplit=1)
                            right_bracket_completion = ")"
                        else:
                            parts = current_line.split(sep=" ", maxsplit=1)
                        if len(parts) == 2:
                            fields_list.append((f"{parts[0]}{right_bracket_completion}".strip(),
                                                parts[1].lstrip()[:parts[1].lstrip().find(" ")]))
                        elif len(parts) == 1:
                            fields_list.append((EMBEDDED_TYPE, parts[0]))
                        elif len(parts) > 2:
                            fields_list.append((f"{parts[0]}{right_bracket_completion}".strip(),
                                                parts[1][:parts[1].find(" ")].strip()))
                        else:
                            pass

                    current_index = current_index + end_of_row + 1
                if type_key.strip() != "" and len(fields_list) != 0:
                    types_mapping[(type_key, the_type.metadata['source'])] = fields_list
            # Primitive type or wrapper of another type
            else:
                types_mapping[(type_key.strip(), the_type.metadata['source'])] = [(type_name.strip(), the_kind.strip())]
        return types_mapping

    def get_function_reserved_word(self) -> str:
        return "func"

    def get_type_reserved_word(self) -> str:
        return "type"

    def is_searchable_file_name(self, function: Document) -> bool:
        file_path = str(function.metadata['source'])
        return "test" not in file_path[file_path.rfind("/") + 1:].split(".")[0].lower()

    def is_function(self, function: Document) -> bool:
        return function.page_content.startswith("func")

    def is_supported_file_extensions(self, extension: str) -> bool:
        return extension in self.supported_files_extensions()

    def supported_files_extensions(self) -> list[str]:
        return [".go"]

    def get_comment_line_notation(self) -> str:
        return "//"

    def dir_name_for_3rd_party_packages(self) -> str:
        return "vendor"

    def is_exported_function(self, function: Document) -> bool:
        function_name = self.get_function_name(function)
        return re.search("[A-Z][a-z0-9-]*", function_name)

    def get_function_name(self, function: Document) -> str:
        try:
            index_of_function_opening = function.page_content.index("{")
        except ValueError as e:
            function_line = function.page_content.find(os.linesep)
            # print(f"function {function.page_content[:function_line]} => contains no body ")
            return function.page_content[:function_line]

        function_header = function.page_content[:index_of_function_opening]
        # function is a method of a type
        if function_header.startswith("func ("):
            index_of_first_right_bracket = function_header.index(")")
            skip_receiver_arg = function_header[index_of_first_right_bracket + 1:]
            index_of_first_left_bracket = skip_receiver_arg.index("(")
            return skip_receiver_arg[:index_of_first_left_bracket].strip()
        # regular function not tied to a certain type
        else:
            try:
                index_of_first_left_bracket = function_header.index("(")
            #    Go Generic function
            except ValueError:
                try:
                    index_of_first_left_bracket = function_header.index("[")
                except ValueError:
                    raise ValueError(f"Invalid function header - {function_header}")
            func_with_name = function_header[:index_of_first_left_bracket]
            if len(func_with_name.split(" ")) > 1:
                return func_with_name.split(" ")[1]
            # TODO Try to extract anonymous function var
            # else:

    def search_for_called_function(self, caller_function: Document, callee_function: str, callee_function_package: str,
                                   code_documents: list[Document], type_documents: list[Document],
                                   callee_function_file_name: str, fields_of_types: dict[tuple, list[tuple]],
                                   functions_local_variables_index: dict[str, dict]) -> bool:
        index_of_function_opening = caller_function.page_content.index("{")
        index_of_function_closing = caller_function.page_content.rfind("}")
        caller_function_body = str(
            caller_function.page_content[index_of_function_opening + 1: index_of_function_closing])
        re.search("", caller_function_body)
        regex = fr'[a-zA-Z0-9_\[\]\(\).]*.?{callee_function}\('
        matching = re.search(regex, caller_function_body, re.MULTILINE)
        if matching and matching.group(0):
            return self.__check_identifier_resolved_to_callee_function_package(function=caller_function,
                                                                               identifier_function=matching.group(0),
                                                                               callee_package=callee_function_package,
                                                                               code_documents=code_documents,
                                                                               caller_function_body=caller_function_body,
                                                                               type_documents=type_documents,
                                                                               callee_function_file_name=
                                                                               callee_function_file_name,
                                                                               fields_of_types=
                                                                               fields_of_types,
                                                                               functions_local_variables_index=
                                                                               functions_local_variables_index)
        else:
            return False

    def __check_identifier_resolved_to_callee_function_package(self, function: Document, identifier_function: str,
                                                               callee_package: str, code_documents: dict[str, Document],
                                                               caller_function_body: str,
                                                               type_documents: list[Document],
                                                               callee_function_file_name: str,
                                                               fields_of_types: dict[tuple, list[tuple]],
                                                               functions_local_variables_index: dict[
                                                                   str, dict]) -> bool:
        caller_function_file = function.metadata.get('source')
        caller_function_name = self.get_function_name(function)
        caller_function_index = f"{caller_function_name}@{caller_function_file}"
        parts = identifier_function.split(".")
        # If there is no qualifier identifier , then check whether callee function and caller
        # function are in the same package
        if len(parts) == 1:
            callee_function = code_documents[callee_function_file_name]
            caller_function = code_documents[function.metadata.get('source')]
            callee_function_package = get_package_name_file(callee_function).strip()
            caller_function_package = get_package_name_file(caller_function).strip()
            if (callee_package in self.get_package_names(function) and
                    callee_function_package == caller_function_package):
                return True
        # There is a qualifier identifier.
        else:
            identifier = parts[-2]
            if identifier.startswith("return"):
                identifier = identifier.replace("return", " ").strip()
            if "(" in identifier:
                identifier = identifier[identifier.index("(") + 1:]

            # verify that identifier resolves to the package name. if identifier is imported in same file, and if so ,
            # if it's the same as callee package name
            # TODO  - reduce list to only files in package
            for doc in code_documents:
                if function.metadata.get('source') == code_documents[doc].metadata.get('source'):
                    # maybe identifier is the package itself in the file
                    regex = f"package {identifier}"
                    code_content = code_documents[doc].page_content
                    matching = re.search(regex, code_content, re.MULTILINE)
                    if matching and matching.group(0):
                        return True

                    identifier_is_imported = handle_imports(code_content, identifier, callee_package)
                    if identifier_is_imported:
                        return True

            ## otherwise, the identifier is in the caller function body or in signature/receiver function argument.

            else:
                # TODO check if identifier is defined in the same
                #  function and dig into structures and identifiers defined by variables
                function_header = function.page_content[:function.page_content.index("{")]
                regex_arguments = r"\([a-zA-Z0-9\s*,.]+\)"
                match_regex = re.finditer(rf"^(\s*|\n){identifier}\s*(:=|=)\s*[^=]+\n*$",
                                          caller_function_body, flags=re.MULTILINE)
                matches = [match.group(0) for match in match_regex]

                if len(matches) > 0:
                    # match_variable = matches[-1]
                    # split = match_variable.split(":=")
                    # if split[0] == match_variable:
                    #     split = match_variable.split("=")
                    # if len(split) > 1:
                    return self.__trace_down_package(expression=identifier.strip(), code_documents=code_documents,
                                                     type_documents=type_documents, callee_package=callee_package,
                                                     fields_of_types=fields_of_types,
                                                     functions_local_variables_index=functions_local_variables_index,
                                                     caller_function_index=caller_function_index)


                # Checks if match some argument in function or receiver parameter ( without parenthesis of return
                # values)
                elif re.search(regex_arguments, function_header):
                    return self.__trace_down_package(expression=identifier.strip(), code_documents=code_documents,
                                                     type_documents=type_documents, callee_package=callee_package,
                                                     fields_of_types=fields_of_types,
                                                     functions_local_variables_index=functions_local_variables_index,
                                                     caller_function_index=caller_function_index)
                # parameters = [tuple(e.replace(")", "").replace("(", "").split(",")) for e
                #               in re.findall(regex_arguments, function_header)[:2]]
                # return check_types_from_callee_package(parameter=identifier, params=parameters,
                #                                        type_documents=type_documents,
                #                                        callee_package=callee_package,
                #                                        code_documents=code_documents,
                #                                        callee_function_file_name=callee_function_file_name
                #                                        )

        return False

    def get_package_names(self, function: Document) -> list[str]:
        package_names = list()
        full_doc_path = str(function.metadata['source'])
        parts = full_doc_path.split("/")
        version = ""
        if len(parts) > 4:
            match = re.search(r"[vV][0-9]{1,2}", parts[4])
            if match and match.group(0):
                version = f"/{match.group(0)}"

        if parts[0].startswith(self.dir_name_for_3rd_party_packages()):
            package_names.append(f"{parts[1]}/{parts[2]}{version}")
            package_names.append(f"{parts[1]}/{parts[2]}/{parts[3]}{version}")
        else:
            try:
                package_names.append(f"{parts[0]}/{parts[1]}{version}")
                package_names.append(f"{parts[0]}/{parts[1]}/{parts[2]}{version}")
            except IndexError as index_excp:
                pass

        return package_names

    def get_package_name(self, function: Document, package_name: str) -> str:
        package_names = self.get_package_names(function)
        for package in package_names:
            if package_name.lower() in package.lower() and package_name.lower() == package.lower():
                return package.lower()
        return None

    def is_root_package(self, function: Document) -> bool:
        return not function.metadata['source'].startswith(self.dir_name_for_3rd_party_packages())

    def is_comment_line(self, line: str) -> bool:
        return line.strip().startswith("//")
