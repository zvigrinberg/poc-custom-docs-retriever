from functions_parsers.golang_functions_parsers import GoLanguageFunctionsParser
from functions_parsers.lang_functions_parsers import LanguageFunctionsParser
from utils.dep_tree import Ecosystem


def get_language_function_parser(ecosystem: Ecosystem) -> LanguageFunctionsParser:
    if ecosystem == Ecosystem.GO:
        return GoLanguageFunctionsParser()
    else:
        return LanguageFunctionsParser()
