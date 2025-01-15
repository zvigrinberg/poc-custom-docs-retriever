import logging
from typing import List, Any, Tuple

import esprima
from langchain_community.document_loaders.parsers.language.javascript import JavaScriptSegmenter

logger = logging.getLogger(f"morpheus.{__name__}")


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

    def extract_functions_classes(self) -> List[str]:
        """Extract functions, classes and exports from the code."""
        if self.skip_file:
            return []

        tree = self._parse_with_fallback()
        if tree is None:
            return []

        functions_classes = []
        for node in tree.body:
            # Handle direct function/class declarations
            if isinstance(node, (esprima.nodes.FunctionDeclaration, esprima.nodes.ClassDeclaration)):
                functions_classes.append(self._extract_code(node))
            # Handle exported declarations
            elif isinstance(node, esprima.nodes.ExportNamedDeclaration):
                if isinstance(node.declaration, (esprima.nodes.FunctionDeclaration, esprima.nodes.ClassDeclaration)):
                    functions_classes.append(self._extract_code(node))

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
