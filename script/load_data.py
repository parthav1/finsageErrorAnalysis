"""
Build the local RAG database: Chroma (chunk embeddings + title summaries) + BM25 index.

Requires JSON files in --source-dir matching the format expected by import_collection_from_dir:
  - Each .json file is a JSON array (.[]).
  - documents[0] is a JSON object string with keys: start, end, date_published (page window).
  - documents[1:] are JSON object strings with at least: content, page_number;
    optional: bundle_id, title_summary.

Example (run from repo root or from script/):

  python script/load_data.py --persist-directory D:/data/my_rag_db --source-dir D:/data/raw_json_lotus

Uses embeddings_model_name from config/example.yaml or config/production.yaml (embedding model
downloads on first run; needs disk space and time).

Collection name defaults to \"lotus\" — same as experiments/retriever/run_*.py.
"""

import argparse
import os
import sys
import logging

logging.basicConfig(
    filename="load_data.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import yaml
import json
from tqdm import tqdm
import shutil
import hashlib
from langchain_community.document_loaders import JSONLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.ragManager import RAGManager
from src.utils.bm25Retriever import load_from_chroma_and_save


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def import_collection_from_dir(rag_manager, collection_name: str, dir_path: str, batch_size: int, ignore_range: bool = False):
    """Load data from a directory into a chroma collection, processing metadata and content.

    Args:
        collection_name: Name of the collection to populate
        dir_path: Path to directory containing JSON files
        ignore_range: Whether to ignore page range restrictions when loading
    """
    chroma, ts_chroma = rag_manager._collections[collection_name]
    chroma.reset_collection()
    ts_chroma.reset_collection()

    content_dict = {}  # Maps content hash to a tuple of (content, metadata)
    gid = 0
    title_summaries = set()

    def hash_content(content: str) -> str:
        """Generate a SHA-256 hash of the content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    filenames = os.listdir(dir_path)
    for filename in filenames:
        if filename.endswith(".json"):
            json_file = os.path.join(dir_path, filename)
            print(json_file)
            loader = JSONLoader(file_path=json_file, jq_schema=".[]", text_content=False)
            documents = loader.load()

            page_range = json.loads(documents[0].page_content)
            print(page_range)
            page_start = page_range["start"]
            page_end = page_range["end"]
            page_date_published = page_range["date_published"]
            count = 0
            for doc in documents[1:]:
                content_dict_data = json.loads(doc.page_content)
                content = content_dict_data.get("content", "")
                page_number = content_dict_data.get("page_number")
                bundle_id = content_dict_data.get("bundle_id", None)
                title_summary = content_dict_data.get("title_summary", None)

                content_hash = hash_content(content)
                if int(page_start) <= int(page_number) <= int(page_end) or ignore_range:
                    metadata = {
                        "filename": filename,
                        "page_number": page_number,
                        "date_published": page_date_published,
                        "doc_id": content_hash,
                        "global_id": gid,
                    }
                    gid += 1
                    if bundle_id:
                        metadata["bundle_id"] = bundle_id
                    if title_summary:
                        metadata["title_summary"] = title_summary
                        title_summaries.add(title_summary)

                    if content_hash in content_dict:
                        existing_content, existing_metadata = content_dict[content_hash]
                        if page_date_published > existing_metadata["date_published"]:
                            content_dict[content_hash] = (content, metadata)
                            logger.debug(
                                f"Replacing content file: {existing_metadata['filename']} page: {existing_metadata['page_number']} in {existing_metadata['date_published']} with new version file: {metadata['filename']} page: {metadata['page_number']} in {page_date_published}. Hash: {content_hash}"
                            )
                    else:
                        content_dict[content_hash] = (content, metadata)

                    count += 1

            logger.info(f"{count} chunks processed in {json_file}.")
    logger.info(f"{len(content_dict)} unique chunks loaded in total.")

    title_summaries = list(title_summaries)
    for i in tqdm(range(0, len(title_summaries), batch_size), desc="Storing title summaries"):
        ts_chroma.add_texts(texts=title_summaries[i : i + batch_size])
    logger.info(f"{len(title_summaries)} title summaries stored in ts_{collection_name} collection.")

    content_list = [item[0] for item in content_dict.values()]
    metadata_list = [item[1] for item in content_dict.values()]
    content_hashes_list = [metadata["doc_id"] for metadata in metadata_list]

    for i in range(len(metadata_list)):
        if i > 0 and metadata_list[i]["filename"] == metadata_list[i - 1]["filename"]:
            metadata_list[i]["prev_chunk_id"] = content_hashes_list[i - 1]
        else:
            metadata_list[i]["prev_chunk_id"] = ""
        if i < len(metadata_list) - 1 and metadata_list[i]["filename"] == metadata_list[i + 1]["filename"]:
            metadata_list[i]["next_chunk_id"] = content_hashes_list[i + 1]
        else:
            metadata_list[i]["next_chunk_id"] = ""

    for i in tqdm(range(0, len(content_list), batch_size), desc="Storing database"):
        batch_contents = content_list[i : i + batch_size]
        batch_metadata = metadata_list[i : i + batch_size]
        batch_doc_ids = content_hashes_list[i : i + batch_size]
        chroma.add_texts(texts=batch_contents, metadatas=batch_metadata, ids=batch_doc_ids)

    logger.info(f"Database stored successfully in {collection_name} collection.")


def main():
    parser = argparse.ArgumentParser(description="Build Chroma + BM25 index from JSON corpus.")
    parser.add_argument(
        "--persist-directory",
        type=str,
        required=True,
        help="Output root folder (will contain chroma/, ts_chroma/, bm25_index/). Must be empty or you accept it being deleted.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Folder containing .json files in the format described in this script's docstring.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="lotus",
        help="Chroma collection name; must match retriever scripts (default: lotus).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML with embeddings_model_name (default: config/production.yaml or config/example.yaml).",
    )
    parser.add_argument(
        "--ignore-range",
        action="store_true",
        help="Include chunks even when page_number is outside the first row's start/end range.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    persist = os.path.abspath(os.path.expanduser(args.persist_directory))
    source_dir = os.path.abspath(os.path.expanduser(args.source_dir))

    if not os.path.isdir(source_dir):
        print(f"Error: --source-dir is not a directory: {source_dir}", file=sys.stderr)
        sys.exit(1)

    if args.config:
        cfg_path = args.config
        if not os.path.isfile(cfg_path):
            print(f"Error: config file not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)
    else:
        cfg_path = None
        for candidate in (
            os.path.join(project_root, "config", "production.yaml"),
            os.path.join(project_root, "config", "example.yaml"),
        ):
            if os.path.isfile(candidate):
                cfg_path = candidate
                break
        if not cfg_path:
            print("Error: no config/production.yaml or config/example.yaml found. Pass --config.", file=sys.stderr)
            sys.exit(1)

    config = load_config(cfg_path)
    config["persist_directory"] = persist
    logger.info("Using config file: %s", cfg_path)
    logger.info("persist_directory: %s", persist)

    if os.path.exists(config["persist_directory"]):
        shutil.rmtree(config["persist_directory"])
        logger.info("Removed existing persist_directory.")

    rag = RAGManager(config)

    collection_name = args.collection_name
    logger.info("Importing collection %s from %s.", collection_name, source_dir)

    rag.create_collection(collection_name)
    import_collection_from_dir(rag, collection_name, source_dir, args.batch_size, ignore_range=args.ignore_range)

    documents = rag.get_collection_documents(collection_name)
    bm25_save_dir = os.path.join(config["persist_directory"], "bm25_index", collection_name)
    load_from_chroma_and_save(documents, bm25_save_dir)
    logger.info("BM25 index saved to %s", bm25_save_dir)
    print(f"Done. Database root: {persist}\nUse retriever scripts with: --persist-directory \"{persist}\"")


if __name__ == "__main__":
    main()
