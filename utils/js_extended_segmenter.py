import logging
import re
from typing import List, Any, Tuple

import esprima
from langchain_community.document_loaders.parsers.language.javascript import JavaScriptSegmenter

from utils.segmenters_utils import get_current_block

logger = logging.getLogger(f"poc.{__name__}")

CONTAINING_SCOPE_SYMBOL = "*%"


def extract_methods_from_objects(code: str) -> list[str]:
    # Regex to get all startings of objects in source code.
    regex = r"(((const|let|var)\s{1,2}[a-zA-Z0-9_-]+\s*=\s*{)|^[a-zA-Z0-9_-]+\s*=\s*{)"
    # methods in objects are tokens that starts with "methodName: function (.*) {" or "methodName(.*) {"
    methods_in_object_regex = (r"([a-zA-Z0-9_-]+:\s*function\s*[(]\s*.*\s*[)]\s*{|^\s*[a-zA-Z0-9_-]+\s*[(]\s*.*\s*["
                               r")]\s*{)")
    matches = re.finditer(regex, code)
    methods = list()
    for matchNum, match in enumerate(matches, start=1):
        current_object = get_current_block(code, match)
        parsed_methods = extract_methods_from_object(current_object=current_object,
                                                     methods_regex=methods_in_object_regex,
                                                     containing_scope=match.group(0))
        methods.extend(parsed_methods)

    return methods


def extract_methods_from_object(current_object: str, methods_regex: str, containing_scope: str) -> list[str]:
    matches = re.finditer(methods_regex, current_object)
    methods = list()
    for matchNum, match in enumerate(matches, start=1):
        current_method = get_current_block(current_object, match)
        methods.append(f"{CONTAINING_SCOPE_SYMBOL}{containing_scope} \n{current_method}")
    return methods


def extract_methods_from_classes(code: str) -> list[str]:
    regex = r"class [a-zA-Z]+ (extends [a-zA-Z0-9]+\s*)?{"
    methods_in_class_regex = r"(get\s)?[a-zA-Z0-9_-]+\s*[(]\s*.*\s*[)]\s*{"
    matches = re.finditer(regex, code)
    methods = list()
    for matchNum, match in enumerate(matches, start=1):
        current_class = get_current_block(code, match)
        parsed_methods = extract_methods_from_object(current_object=current_class, methods_regex=methods_in_class_regex,
                                                     containing_scope=match.group(0))
        methods.extend(parsed_methods)

    return methods


def get_all_functions_global_scope(code: str) -> list[str]:
    regex = r"^(export)?\s?(default)?\s?(async)?\s?function\s{1,2}[a-z-A-Z0-9_-]+[(].*[)]\s*{"
    functions = get_all_functions(code, regex)

    return functions


def extract_anonymous_functions(code: str) -> list[str]:
    # Capture all anonymous functions in outer scope in the source code module
    regex = r"^((const|let|var)\s{1,2})?[a-zA-Z0-9_-]+\s*=\s*function\s*[(].*[)]\s*{"
    functions = get_all_functions(code, regex)

    return functions


def extract_lambda_functions(code: str) -> list[str]:
    regex = r"^((const|let|var)\s{1,2})?[a-zA-Z0-9_-]+\s*=\s*([(].*[)]|a-zA-Z0-9_-]*)\s*=>\s*.*"
    functions = list()
    matches = re.finditer(regex, code, flags=re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        if match.group(0).__contains__("{"):
            current_lambda_function = get_current_block(code, match)
        else:
            current_lambda_function = match.group(0)

        functions.append(current_lambda_function.strip())

    return functions


# extract_lambda_functions("const functionName = (params,b) => { blabla = params +  kars; \n blabla = blabla*2; \n return blabla;  } \nconst add = (a,b) => a + b \n")


def get_all_functions(code, regex):
    matches = re.finditer(regex, code, flags=re.MULTILINE)
    functions = list()
    for matchNum, match in enumerate(matches, start=1):
        current_function = get_current_block(code, match)
        functions.append(current_function)
    return functions


class ExtendedJavaScriptSegmenter(JavaScriptSegmenter):
    """Extended JavaScript segmenter that handles shebang and ES optional chaining."""

    def __init__(self, code: str):
        """Initialize the segmenter with preprocessed code."""
        super().__init__(code)

        if code.startswith("#!"):
            logger.warning("File contains a shebang line. Skipping parsing.")
            self.skip_file = True
        else:
            self.skip_file = False
            self.code = self.code.replace("?.", ".")

    def _parse_with_fallback(self) -> Any:
        """Try to parse code as script first, then as module if that fails."""
        try:
            logger.debug("Attempting to parse as a script...")
            return esprima.parseScript(self.code, loc=True)
        except esprima.Error:
            logger.debug("Script parsing failed. Trying module parsing...")
            try:
                return esprima.parseModule(self.code, loc=True)
            except esprima.Error as e:
                logger.error("Module parsing failed: %s", str(e))
                return None

    def __extract_all_methods(self) -> List[str]:
        # each function will start with comment of the containing scope.
        methods_from_objects = extract_methods_from_objects(self.code)
        methods_from_classes = extract_methods_from_classes(self.code)
        anonymous_functions = extract_anonymous_functions(self.code)
        lambda_functions = extract_lambda_functions(self.code)
        return [*methods_from_objects, *methods_from_classes, *anonymous_functions, *lambda_functions]

    def extract_functions_classes(self) -> List[str]:
        """Extract functions, classes and exports from the code."""
        if self.skip_file:
            return []

        tree = self._parse_with_fallback()
        functions_classes = []
        if tree is None:
            # extract all regular functions
            functions_classes = get_all_functions_global_scope(self.code)
        else:
            for node in tree.body:
                # Handle direct function/class declarations
                if isinstance(node, (esprima.nodes.FunctionDeclaration, esprima.nodes.ClassDeclaration)):
                    functions_classes.append(self._extract_code(node))
                # Handle exported declarations
                elif isinstance(node, esprima.nodes.ExportNamedDeclaration):
                    if isinstance(node.declaration,
                                  (esprima.nodes.FunctionDeclaration, esprima.nodes.ClassDeclaration)):
                        functions_classes.append(self._extract_code(node))

        methods = self.__extract_all_methods()
        functions_classes.extend(methods)
        return functions_classes

    def simplify_code(self) -> str:
        """Simplify the code by replacing function/class bodies with comments."""
        if self.skip_file:
            return self.code

        tree = self._parse_with_fallback()
        if tree is None:
            return self.code

        simplified_lines = self.source_lines[:]
        indices_to_del: List[Tuple[int, int]] = []

        for node in tree.body:
            if isinstance(node, (esprima.nodes.FunctionDeclaration, esprima.nodes.ClassDeclaration)):
                start, end = node.loc.start.line - 1, node.loc.end.line
                simplified_lines[start] = f"// Code for: {simplified_lines[start]}"
                indices_to_del.append((start + 1, end))
            elif isinstance(node, esprima.nodes.ExportNamedDeclaration):
                if isinstance(node.declaration, (esprima.nodes.FunctionDeclaration, esprima.nodes.ClassDeclaration)):
                    start, end = node.loc.start.line - 1, node.loc.end.line
                    simplified_lines[start] = f"// Code for: {simplified_lines[start]}"
                    indices_to_del.append((start + 1, end))

        for start, end in reversed(indices_to_del):
            del simplified_lines[start:end]

        return "\n".join(line for line in simplified_lines)
