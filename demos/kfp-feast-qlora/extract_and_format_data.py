import os
import json
import pickle
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import Dict, List, Tuple

OUTFILE = "labeled_CISI_IR_data.pickle"
DATA_PATH = "nlp_retrieval_dataset/"
SHORT_EXAMPLE_INDEX = 111


def parse_documents(file_path: str) -> Dict[int, str]:
    with open(file_path, "r") as file:
        lines = ""
        for line in file:
            lines += "\n" + line.strip() if line.startswith(".") else " " + line.strip()
        lines = lines.lstrip("\n").split("\n")

    documents = {}
    doc_id = 0
    doc_text = ""
    for line in lines:
        if line.startswith(".I"):
            if doc_id != 0:
                documents[doc_id] = doc_text.lstrip(" ")
            doc_id = int(line.split(" ")[1].strip())
            doc_text = ""
        elif line.startswith(".X"):
            if doc_id != 0:
                documents[doc_id] = doc_text.lstrip(" ")
                doc_id = 0
        else:
            doc_text += (
                line[3:].strip() + " "
            )  # Ignore the first 3 characters of each line.

    if doc_id != 0:
        documents[doc_id] = doc_text.lstrip(" ")

    return documents


def parse_queries(file_path: str) -> Dict[int, str]:
    with open(file_path, "r") as file:
        lines = ""
        for line in file:
            lines += "\n" + line.strip() if line.startswith(".") else " " + line.strip()
        lines = lines.lstrip("\n").split("\n")

    queries = {}
    qry_id = 0
    for line in lines:
        if line.startswith(".I"):
            qry_id = int(line.split(" ")[1].strip())
        elif line.startswith(".W") and qry_id != 0:
            queries[qry_id] = line[3:].strip()  # The actual query text follows ".W".
            qry_id = 0

    return queries


def parse_relevance(file_path: str) -> Dict[int, List[int]]:
    relevance = {}
    with open(file_path, "r") as f:
        for line in f:
            qry_id = int(line.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0])
            doc_id = int(line.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1])
            if qry_id in relevance:
                relevance[qry_id].append(doc_id)
            else:
                relevance[qry_id] = []
                relevance[qry_id].append(doc_id)

    return relevance


def read_data(
    all_file_path: str, query_file_path: str, rel_file_path: str
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, List[int]]]:
    doc_set = parse_documents(all_file_path)
    qry_set = parse_queries(query_file_path)
    rel_set = parse_relevance(rel_file_path)

    print(
        f"""
# of documents          = {len(doc_set)};
# of queries            = {len(qry_set)};
# of related documents  = {len(rel_set)};
        """
    )
    return doc_set, qry_set, rel_set


def format_single_article_into_messages(
    queries: Dict, documents: Dict, related_docs: Dict, query_id: int
):
    system_prompt = f"""
Please use the article below to either respond to the query or answer the question if applicable. Don't mention the article title and please be brief.

--- Begin article ---
{documents[related_docs[query_id][0]]}
--- End article ---

"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{queries[query_id]}"},
    ]
    return messages


def format_all_articles_into_messages(
    queries: Dict, documents: Dict, related_docs: Dict, query_id: int, n: int = None
):
    if n:
        relevant_docs = "\n\n".join(
            f"Article {i}: {documents[qid]}"
            for i, qid in enumerate(related_docs[query_id])
            if (i + 1) <= n_articles
        )
    else:
        # Swap i for qid if you want the actual id, i is just a counter
        relevant_docs = "\n\n".join(
            f"Article {i}: {documents[qid]}"
            for i, qid in enumerate(related_docs[query_id])
        )
    system_prompt = f"""
Please use the article(s) below to either respond to the query or answer the question if applicable. Don't mention the article title and please be brief.

--- Begin articles ---
{relevant_docs} -- note the deleted "answer" key
--- End articles ---

"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{queries[qidx]}"},
    ]
    return messages


def load_raw_data() -> Dict:
    for dirname, _, filenames in os.walk(DATA_PATH):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    d, q, r = read_data(
        "nlp_retrieval_dataset/CISI.ALL",
        "nlp_retrieval_dataset/CISI.QRY",
        "nlp_retrieval_dataset/CISI.REL",
    )

    sample_message = format_single_article_into_messages(q, d, r, 1)
    print(json.dumps(sample_message[SHORT_EXAMPLE_INDEX], indent=2))

    message_data = {}
    for query_id in q:
        try:
            message_data[query_id] = {
                "messages": format_all_articles_into_messages(q, d, r, query_id)
            }
        except:
            pass
    return message_data


def load_processed_data() -> Dict:
    print(f"loading processed data with generated answers from {OUTFILE}")
    processed_message_data = pickle.load(open(OUTFILE, "rb"))
    print(json.dumps(processed_message_data[SHORT_EXAMPLE_INDEX], indent=2))
    return processed_message_data


def process_and_save_raw_data(message_data: Dict) -> Dict:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    for query_id in message_data:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini", messages=message_data[query_id]["messages"]
            )
            message_data[query_id]["answer"] = completion.choices[0].message.content
        except:
            print(f"query {query_id} failed")
    with open(OUTFILE, "wb") as f:
        pickle.dump(processed_message_data, f)

    return message_data


def format_data_for_training(message_data: Dict) -> Dict:
    output_data = {}
    for query_id in message_data:
        output_data[query_id] = message_data[query_id]
        output_data[query_id]["messages"].append(
            {"role": "assistant", "content": message_data[query_id]["answer"]}
        )
        del output_data[query_id]["answer"]
    return output_data


def main():
    if OUTFILE in os.listdir():
        processed_message_data = load_processed_data()
    else:
        raw_message_data = load_raw_data()
        processed_message_data = process_and_save_raw_data(raw_message_data)

    formatted_training_data = format_data_for_training(processed_message_data)
    print("*" * 30)
    print("formatted data")
    print("*" * 30)
    print(json.dumps(formatted_training_data[SHORT_EXAMPLE_INDEX], indent=2))


if __name__ == "__main__":
    main()
