import copy
import json
import logging
import os
import sys
import time
import typing
from hashlib import sha512
from pathlib import Path
from pathlib import PurePath

from langchain.docstore.document import Document
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.language.language_parser import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders.parsers.language.code_segmenter import CodeSegmenter
from langchain_community.document_loaders.parsers.language.language_parser import LANGUAGE_EXTENSIONS
from langchain_community.document_loaders.parsers.language.language_parser import LANGUAGE_SEGMENTERS
from langchain_core.document_loaders.blob_loaders import Blob

from data_models.input import SourceDocumentsInfo
from .source_code_git_loader import SourceCodeGitLoader

if typing.TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings  # pragma: no cover

PathLike = typing.Union[str, os.PathLike]

logger = logging.getLogger(f"morpheus.{__name__}")


class MultiLanguageRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """
    A version of langchain's RecursiveCharacterTextSplitter that supports multiple languages.
    """

    def __init__(
            self,
            keep_separator: bool = True,
            **kwargs,
    ) -> None:
        """Create a new RecursiveCharacterTextSplitter."""
        super().__init__(is_separator_regex=True, keep_separator=keep_separator, **kwargs)

    def _get_separators(self, language: Language) -> list[str]:
        try:
            return RecursiveCharacterTextSplitter.get_separators_for_language(language)
        except ValueError:
            return self._separators

    def create_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0

            # Determine the language of the text from the metadata
            language = _metadatas[i].get("language", None)

            for chunk in self._split_text(text, separators=self._get_separators(language)):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents


class ExtendedLanguageParser(LanguageParser):
    """
    A version of langchain's LanguageParser that supports extended file extension and language parsing.
    """

    LANGUAGE_EXTENSIONS: dict[str, str] = {
        **LANGUAGE_EXTENSIONS,
        "h": "c",
        "hpp": "cpp",
    }

    LANGUAGE_SEGMENTERS: dict[str, type[CodeSegmenter]] = {
        **LANGUAGE_SEGMENTERS,
    }

    def lazy_parse(self, blob: Blob) -> typing.Iterator[Document]:
        try:
            code = blob.as_string()
        except Exception as e:
            logger.warning("Failed to read code for '%s'. Ignoring this file. Error: %s", blob.source, e)
            return

        language = self.language or (self.LANGUAGE_EXTENSIONS.get(blob.source.rsplit(".", 1)[-1]) if isinstance(
            blob.source, str) else None)

        if language is None:
            yield Document(
                page_content=code,
                metadata={
                    "source": blob.source,
                },
            )
            return

        if self.parser_threshold >= len(code.splitlines()):
            yield Document(
                page_content=code,
                metadata={
                    "source": blob.source,
                    "language": language,
                },
            )
            return

        segmenter = self.LANGUAGE_SEGMENTERS[language](blob.as_string())

        try:
            extracted_functions_classes = segmenter.extract_functions_classes()

        except Exception as e:

            logger.warning("Failed to parse code for '%s'. Ignoring this file. Error: %s",
                           blob.source,
                           e,
                           exc_info=True)
            extracted_functions_classes = []

        # If the code didnt parse, and there are no functions or classes, return the original code
        if not segmenter.is_valid() and len(extracted_functions_classes) == 0:
            yield Document(
                page_content=code,
                metadata={
                    "source": blob.source,
                    "language": language,
                },
            )
            return

        for functions_classes in extracted_functions_classes:
            yield Document(
                page_content=functions_classes,
                metadata={
                    "source": blob.source,
                    "content_type": "functions_classes",
                    "language": language,
                },
            )

        try:
            simplified_code = segmenter.simplify_code()
        # If simplifying the code fails, return the original code
        except Exception as e:
            logger.warning("Failed to simplify code for '%s'. Returning original code. Error: %s",
                           blob.source,
                           e,
                           exc_info=True)
            yield Document(
                page_content=code,
                metadata={
                    "source": blob.source,
                    "language": language,
                },
            )
        else:
            yield Document(
                page_content=simplified_code,
                metadata={
                    "source": blob.source,
                    "content_type": "simplified_code",
                    "language": language,
                },
            )


class DocumentEmbedding:
    """
    A class to create a FAISS database from a list of source documents. The source documents are collected from git
    repositories and chunked into smaller pieces for embedding.
    """

    def __init__(self,
                 *,
                 embedding: "Embeddings",
                 vdb_directory: PathLike = "./.cache/am_cache/vdb",
                 git_directory: PathLike = "./.cache/am_cache/git",
                 chunk_size: int = 800,
                 chunk_overlap: int = 160):
        """
        Create a new DocumentEmbedding instance.

        Parameters
        ----------
        embedding : Embeddings
            The embedding to use for the FAISS database.
        vdb_directory : PathLike
            The directory to save the FAISS database. The database will be saved in a subdirectory based on the hash of
            the source documents.
        git_directory : PathLike, optional
            The directory to use for the Git repository cloning, by default "./.tmp/git_cache"
        chunk_size : int, optional
            Maximum size of a single chunk, by default 1000
        chunk_overlap : int, optional
            Overlap between chunks, by default 200
        """

        self._embedding = embedding
        self._vdb_directory = Path(vdb_directory)
        self._git_directory = Path(git_directory)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def embedding(self):
        return self._embedding

    @property
    def vdb_directory(self):
        return self._vdb_directory

    @property
    def git_directory(self):
        return self._git_directory

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def chunk_overlap(self):
        return self._chunk_overlap

    def hash_source_documents_info(self, source_infos: list[SourceDocumentsInfo]):
        """
        Hash the source documents info to create a unique hash for the source documents. This is used to determine if
        the VDB already exists for the source documents.

        Parameters
        ----------
        source_infos : list[SourceDocumentsInfo]
            A list of source documents info to hash.

        Returns
        -------
        str
            Returns a unique hash for the source documents as a positive integer. Will always return the same hash for
            the same source regardless of order.
        """

        obj_to_hash = {
            "class": self.__class__.__qualname__,
            "source_infos": [hash(x) for x in sorted(source_infos)],
            "embedding": str(self._embedding),
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
        }

        # Hash the source documents info
        hash_val = int.from_bytes(bytes=sha512(f"{json.dumps(obj_to_hash)}".encode('utf-8', errors='ignore')).digest(),
                                  byteorder=sys.byteorder)

        # Convert to a positive, consistent hash (Avoids negative hex values)
        nbits = 64
        return hex((hash_val + (1 << nbits)) % (1 << nbits))[2:]  # Skip the '0x' prefix

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Chunk the documents into smaller pieces for embedding.
        """

        splitter = MultiLanguageRecursiveCharacterTextSplitter(chunk_size=self._chunk_size,
                                                               chunk_overlap=self._chunk_overlap)
        split_documents = splitter.split_documents(documents)

        return split_documents

    def get_repo_path(self, source_info: SourceDocumentsInfo):
        """
        Returns the path to the git repository for a source document info.

        Parameters
        ----------
        source_info : SourceDocumentsInfo
            The source document info to get the git repository path.

        Returns
        -------
        Path
            Returns the path to the git repository.
        """
        return self._git_directory / PurePath(source_info.git_repo)

    def collect_documents(self, source_info: SourceDocumentsInfo) -> list[Document]:
        """
        Collect documents from a source document info. This will clone the git repository and collect files from the
        repository based on the include and exclude patterns. Each file is then parsed and segmented based on its
        language into Documents.

        Parameters
        ----------
        source_info : SourceDocumentsInfo
            The source document info to collect documents

        Returns
        -------
        list[Document]
            Returns a list of documents collected from the source document info.
        """

        repo_path = self.get_repo_path(source_info)

        blob_loader = SourceCodeGitLoader(repo_path=repo_path,
                                          clone_url=source_info.git_repo,
                                          ref=source_info.ref,
                                          include=source_info.include,
                                          exclude=source_info.exclude)

        blob_parser = ExtendedLanguageParser()

        loader = GenericLoader(blob_loader=blob_loader, blob_parser=blob_parser)

        documents = loader.load()

        logger.debug("Collected documents for '%s', Document count: %d", repo_path, len(documents))

        return documents

    def create_vdb(self, source_infos: list[SourceDocumentsInfo], output_path: PathLike):
        """
        Create a FAISS database from a list of input directories.

        Parameters
        ----------
        source_infos : list[SourceDocumentsInfo]
            A list of source documents to create the VDB from. All documents will be collected from the source
            documents.
        output_path : PathLike
            The location to save the FAISS database.

        Returns
        -------
        FAISS
            Returns an instance of the FAISS database.
        """

        logger.debug("Collecting documents from git repos. Source Infos: %s",
                     json.dumps([x.model_dump(mode="json") for x in source_infos]))

        output_path = Path(output_path)

        # Warn if the output path already exists and we will overwrite it
        if (output_path.exists()):

            logger.warning("Vector Database already exists and will be overwritten: %s", output_path)

        documents = []
        for input_dir in source_infos:
            documents.extend(self.collect_documents(input_dir))

        # Apply chunking on the source documents
        chunked_documents = self._chunk_documents(documents)

        logger.debug("Creating FAISS database from source documents. Doc count: %d, Chunks: %s, Location: %s",
                     len(documents),
                     len(chunked_documents),
                     output_path)

        saved_show_progress = getattr(self._embedding, "show_progress", None)
        saved_show_progress_bar = getattr(self._embedding, "show_progress_bar", None)

        # Attempt to enable progress bar for VDB generation if supported
        if (hasattr(self._embedding, "show_progress")):
            setattr(self._embedding, "show_progress", True)
        if (hasattr(self._embedding, "show_progress_bar")):
            setattr(self._embedding, "show_progress_bar", True)

        embedding_start_time = time.time()

        # Create the FAISS database
        db = FAISS.from_documents(chunked_documents, self._embedding)

        logger.debug("Completed embedding in %.2f seconds for '%s'", time.time() - embedding_start_time, output_path)

        # Clear the CUDA cache if torch is available. This is to prevent the pytorch cache from growing when it will not be reused
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

        # Revert progress bar setting
        if (saved_show_progress is not None):
            setattr(self._embedding, "show_progress", saved_show_progress)
        if (saved_show_progress_bar is not None):
            setattr(self._embedding, "show_progress_bar", saved_show_progress_bar)

        # Ensure the directory exists
        output_path.mkdir(exist_ok=True, parents=True)

        # Save the database
        db.save_local(str(output_path))

        return db

    def build_vdbs(self,
                   input_sources: list[SourceDocumentsInfo],
                   ignore_code_embedding: bool = False) -> tuple[Path | None, Path | None]:
        """
        Build the code and document VDB based on a list of source documents.

        Parameters
        ----------
        input_sources : list[SourceDocumentsInfo]
            A list of source documents to create the VDB from.

        Returns
        -------
        tuple[Path | None, Path | None]
            Returns a tuple of (code_vdb, doc_vdb). If the VDB was not created, the value will be None for each type.
        """
        code_vdb: Path | None = None
        doc_vdb: Path | None = None

        # Create embeddings for each source type
        for source_type in ["code", "doc"]:

            if ignore_code_embedding and source_type == "code":
                continue

            # Filter the source documents
            source_infos = [source_info for source_info in input_sources if source_info.type == source_type]

            if source_infos:

                # Determine the output path by combining the vdb_directory with the hash of the source documents
                vdb_output_dir = self.vdb_directory / source_type / str(self.hash_source_documents_info(source_infos))

                if (not vdb_output_dir.exists() or os.environ.get("MORPHEUS_ALWAYS_REBUILD_VDB", "0") == "1"):
                    vdb = self.create_vdb(source_infos=source_infos, output_path=vdb_output_dir)
                else:
                    logger.debug("Cache hit on VDB. Loading existing FAISS database: %s", vdb_output_dir)

                    vdb = FAISS.load_local(str(vdb_output_dir), self._embedding, allow_dangerous_deserialization=True)

            else:
                vdb_output_dir = None
                vdb = None

            if (source_type == "code"):
                if (vdb is not None):
                    code_vdb = vdb_output_dir
            elif (source_type == "doc"):
                if (vdb is not None):
                    doc_vdb = vdb_output_dir
            else:
                raise ValueError(f"Unknown source type: {source_type}")  # pragma: no cover

        return code_vdb, doc_vdb
