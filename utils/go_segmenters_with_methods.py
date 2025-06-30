from typing import List
import re

from langchain_community.document_loaders.parsers.language.go import GoSegmenter

from utils.segmenters_utils import get_current_block


def parse_all_methods(code: str) -> list[str]:
    # regex = r"func\s*\([a-zA-Z0-9\s]*\) [a-zA-X]+\([a-zA-Z0-9\s]*\)([a-zA-Z0-9\s]*){"
    regex = r"func\s*\([a-zA-Z0-9\s\*.]+\) [a-zA-Z]+\([,a-zA-Z0-9\s\[\].]*\)\s*(\(?[a-zA-Z0-9\s.,*]+\)?)?\s*{"
    methods = get_all_functions(code, regex)
    return methods


def parse_all_anonymous_functions(code: str) -> list[str]:
    regex = r"(var)? [a-zA-Z0-9_]+\s*(:)?=\s*func\s*.*{"
    functions = get_all_functions(code, regex)
    return functions


def get_all_functions(code: str, regex: str):
    matches = re.finditer(regex, code)
    functions = list()
    for matchNum, match in enumerate(matches, start=1):
        current_method = get_current_block(code, match)
        functions.append(current_method)
    return functions


class GoSegmenterWithMethods(GoSegmenter):

    def __init__(self, code: str):
        super().__init__(code)

    def extract_functions_classes(self) -> List[str]:
        function_classes = super().extract_functions_classes()
        methods_classes = parse_all_methods(self.code)
        parse_anonymous_functions = parse_all_anonymous_functions(self.code)
        function_classes.extend(methods_classes)
        function_classes.extend(parse_anonymous_functions)
        return function_classes

# test_str = """func (any *uint64Any) LastError() error {
# 	return nil
# }
# """
# parse_all_methods(test_str)
