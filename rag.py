import os
import json
import argparse
import numpy as np
import torch
from torch import Tensor
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from retrievers.e5_mistral import get_e5_mistral_embeddings_for_query, get_e5_mistral_embeddings_for_document
from readers.metrics import ems, f1_score, accuracy
import re
from concurrent.futures import ThreadPoolExecutor

# Global variables
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============ Llama3-8B-Instruct Model Loader ============
def load_llama3_model_tokenizer(model_path):
    """
    Load Llama3-8B-Instruct model and tokenizer

    Args:
        model_path: Model path

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading Llama3-8B-Instruct model: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left"
    )

    # Set pad_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Setting padding token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )
        model.to(device)

    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


# ============ Gemma-7B Model Loader ============
def load_gemma_model_tokenizer(model_path):
    """
    Load Gemma-7B model and tokenizer

    Args:
        model_path: Model path

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading Gemma-7B model: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True
    )

    # Set pad_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Setting padding token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.to(device)

    model.eval()
    print("Gemma model loaded successfully!")
    return model, tokenizer

# ============ Mistral-7B Model Loader ============
def load_mistral_model_tokenizer(model_path):
    """
    Load Mistral-7B model and tokenizer

    Args:
        model_path: Model path

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading Mistral-7B model: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True
    )

    # Set pad_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Setting padding token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.to(device)

    model.eval()
    print("Mistral model loaded successfully!")
    return model, tokenizer

# ============ Similarity-based Retrieval Function ============
def retrieve_documents_by_similarity(question: str, ctxs: List[Dict], args) -> List[Dict]:
    """
    Similarity-based retrieval function: Retrieve the most relevant documents for a single question
    Only uses similarity score, ignores truthful_score

    Args:
        question: Question text
        ctxs: List of candidate documents
        args: Configuration arguments

    Returns:
        List[Dict]: Retrieved top-k documents containing the 'text' field
    """
    # Extract all candidate documents
    documents = []
    end_index = len(ctxs) - 3
    ctxs = ctxs[:end_index + args.fake_num]
    for ctx in ctxs:
        documents.append("title: {}, text: {}".format(ctx["title"], ctx["text"]))

    # Calculate document embeddings
    doc_embeddings = get_e5_mistral_embeddings_for_document(documents, max_length=256, batch_size=2)
    question_embedding = get_e5_mistral_embeddings_for_query("retrieve_relevant_documents", [question],
                                                             max_length=128, batch_size=1)

    # Normalize embeddings
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
    question_embedding = torch.nn.functional.normalize(question_embedding, p=2, dim=-1)

    # Calculate similarity scores
    similarities = torch.matmul(question_embedding, doc_embeddings.T).squeeze(0)

    # Select top-k documents
    topk_scores, topk_indices = torch.topk(similarities, k=min(args.context_nums, len(documents)), dim=0)

    # Build retrieval results
    retrieved_documents = []
    for idx in topk_indices.tolist():
        retrieved_documents.append({
            "text": documents[idx]
        })

    return retrieved_documents


def retrieve_documents_by_similarity_score(question: str, ctxs: List[Dict], args, ideal_setting: bool = False) -> List[
    Dict]:
    """
    Single-hop retrieval function: Retrieve the most relevant documents for a single question
    Uses a scoring method that multiplies similarity by truthful_score

    Args:
        question: Question text
        ctxs: List of candidate documents
        args: Configuration arguments

    Returns:
        List[Dict]: Retrieved top-k documents containing 'text' and 'credibility' fields
    """
    # Extract all candidate documents and truthful scores
    documents, truthful_scores = [], []
    end_index = len(ctxs) - 3
    ctxs = ctxs[:end_index + args.fake_num]
    for i, ctx in enumerate(ctxs):
        documents.append("title: {}, text: {}".format(ctx["title"], ctx["text"]))
        if ideal_setting:
            if i < end_index:
                truthful_scores.append(10)
            else:
                truthful_scores.append(1)
        else:
            # Get truthful_score of the document
            truthful_scores.append(ctx["text_truthful_score"])

    # Convert truthful_scores to tensor
    truthful_scores_tensor = torch.tensor(truthful_scores, dtype=torch.bfloat16)
    # Calculate document embeddings
    doc_embeddings = get_e5_mistral_embeddings_for_document(documents, max_length=256, batch_size=2)
    question_embedding = get_e5_mistral_embeddings_for_query("retrieve_relevant_documents", [question],
                                                             max_length=128, batch_size=1)
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
    question_embedding = torch.nn.functional.normalize(question_embedding, p=2, dim=-1)

    # Calculate similarity scores
    similarities = torch.matmul(question_embedding, doc_embeddings.T).squeeze(0)
    final_scores = similarities * truthful_scores_tensor
    # Select top-k documents
    topk_scores, topk_indices = torch.topk(final_scores, k=min(args.context_nums, len(documents)), dim=0)

    # Build retrieval results, returning only text and truthful_score (credibility) fields
    retrieved_documents = []
    for i, idx in enumerate(topk_indices.tolist()):
        retrieved_documents.append({
            "text": documents[idx],
            "credibility": truthful_scores[idx]
        })
    if not ideal_setting and args.exclusion:
        retrieved_documents = [doc for doc in retrieved_documents if doc["credibility"] > 3]
    return retrieved_documents

def retrieve_documents_by_similarity_andCredibility(question: str, ctxs: List[Dict], args, ideal_setting: bool = True) -> List[
    Dict]:
    """
    Single-hop retrieval function: Retrieve the most relevant documents for a single question
    Uses a weighted sum of similarity and truthful_score

    Args:
        question: Question text
        ctxs: List of candidate documents
        args: Configuration arguments

    Returns:
        List[Dict]: Retrieved top-k documents containing 'text' field
    """
    # Extract all candidate documents and truthful scores
    documents, truthful_scores = [], []
    end_index = len(ctxs) - 3
    ctxs = ctxs[:end_index + args.fake_num]
    for i, ctx in enumerate(ctxs):
        # Use only text content as retrieval document
        documents.append("title: {}, text: {}".format(ctx["title"], ctx["text"]))
        if ideal_setting:
            if i < end_index:
                truthful_scores.append(10)
            else:
                truthful_scores.append(1)
        else:
            # Get truthful_score of the document
            truthful_scores.append(ctx["text_truthful_score"])

    # Convert truthful_scores to tensor
    truthful_scores_tensor = torch.tensor(truthful_scores, dtype=torch.bfloat16)
    # Calculate document embeddings
    doc_embeddings = get_e5_mistral_embeddings_for_document(documents, max_length=256, batch_size=2)
    question_embedding = get_e5_mistral_embeddings_for_query("retrieve_relevant_documents", [question],
                                                             max_length=128, batch_size=1)
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
    question_embedding = torch.nn.functional.normalize(question_embedding, p=2, dim=-1)

    # Calculate similarity scores
    similarities = torch.matmul(question_embedding, doc_embeddings.T).squeeze(0)
    final_scores = 0.4 * similarities + 0.6 * truthful_scores_tensor
    # Select top-k documents
    topk_scores, topk_indices = torch.topk(final_scores, k=min(args.context_nums, len(documents)), dim=0)

    # Build retrieval results
    retrieved_documents = []
    for idx in topk_indices.tolist():
        retrieved_documents.append({
            "text": documents[idx]
        })
    return retrieved_documents

# ============ Llama3 Data Processor ============
class Llama3DataProcessor:
    """Llama3 Data Processor"""

    def __init__(self, args):
        self.args = args

    def get_contexts(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted context string
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            contexts.append(("Passage-%d: " % i) + text)

        return "\n".join(contexts)

    def get_contexts_score(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context
        Unlike get_contexts, this method adds a truthful score after each Passage
        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted context string
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            score = doc["credibility"]
            contexts.append(("Passage-%d: " % i) + text + (f'\nTruthful score: {score}\n'))

        return "\n".join(contexts)

    def create_llama3_prompt(self, question: str, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        Create a prompt formatted for Llama3

        Args:
            question: Question text
            retrieved_docs: List of retrieved documents

        Returns:
            List[Dict]: Chat-formatted prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts(retrieved_docs)
            prompt = f'Given the following information: \n{contexts}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source.\nQuestion: {question}\nAnswer: '
        else:
            print("norag")
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nAnswer: '

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant!"
            },
            {
                "role": "user",
                "content": prompt
            },
        ]

        return messages

    def create_llama3_prompt_score(self, question: str, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        Create a prompt formatted for Llama3 with scores

        Args:
            question: Question text
            retrieved_docs: List of retrieved documents

        Returns:
            List[Dict]: Chat-formatted prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts_score(retrieved_docs)
            prompt = f'You are an assistant capable of answering questions based on provided passages. Each passage is assigned a truthfulness score (0-10), where a higher score indicates greater credibility. Your answer need to combine multiple passages and their credibility. Given the following information: \n{contexts}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source (just output a answer, don\'t output anything else).\nQuestion: {question}\nAnswer: '
        else:
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nAnswer: '

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant!"
            },
            {
                "role": "user",
                "content": prompt
            },
        ]

        return messages


# ============ Gemma Data Processor ============
class GemmaDataProcessor:
    """Gemma Data Processor"""

    def __init__(self, args):
        self.args = args

    def get_contexts(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted context string
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            contexts.append(("Passage-%d: " % i) + text)

        return "\n".join(contexts)

    def get_contexts_score(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context
        Unlike get_contexts, this method adds a truthful score after each Passage
        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted context string
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            score = doc["credibility"]
            contexts.append(("Passage-%d: " % i) + text + (f'\nTruthful score: {score}\n'))

        return "\n".join(contexts)

    def create_gemma_prompt(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        Create a prompt formatted for Gemma

        Args:
            question: Question text
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts(retrieved_docs)
            prompt = f'Given the following information: \n{contexts}\nplease only output the answer to the question.\nQuestion:{question}\nthe correct answer is:'
        else:
            print("norag")
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nthe correct answer is:'

        return prompt

    def create_gemma_prompt_score(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        Create a prompt formatted for Gemma with scores

        Args:
            question: Question text
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts_score(retrieved_docs)
            prompt = f'You are an assistant capable of answering questions based on provided passages. Each passage is assigned a truthfulness score (0-10), where a higher score indicates greater credibility. You should consider truthfulness score of the passage, if the score is low, you should not trust it. Given the following information: \n{contexts}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source (just output a answer, don\'t output anything else).\nQuestion: {question}\nthe correct answer is:'
        else:
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nthe correct answer is:'

        return prompt

# ============ Mistral Data Processor ============
class MistralDataProcessor:
    """Mistral Data Processor"""

    def __init__(self, args):
        self.args = args

    def get_contexts(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted context string
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            contexts.append(("Passage-%d: " % i) + text)

        return "\n".join(contexts)

    def get_contexts_score(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context
        Unlike get_contexts, this method adds a truthful score after each Passage
        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted context string
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            score = doc["credibility"]
            contexts.append(("Passage-%d: " % i) + text + (f'\nTruthful score: {score}\n'))

        return "\n".join(contexts)

    def create_mistral_prompt(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        Create a prompt formatted for Mistral

        Args:
            question: Question text
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts(retrieved_docs)
            prompt = f'Given the following information: \n{contexts}\nplease only output the answer to the question.\nQuestion:{question}\nthe correct answer is:'
        else:
            print("norag")
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nthe correct answer is:'

        return prompt

    def create_mistral_prompt_score(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        Create a prompt formatted for Mistral with scores

        Args:
            question: Question text
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts_score(retrieved_docs)
            prompt = f'You are an assistant capable of answering questions based on provided passages. Each passage is assigned a truthfulness score (0-10), where a higher score indicates greater credibility. You should consider truthfulness score of the passage, if the score is low, you should not trust it. Given the following information: \n{contexts}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source (just output a answer, don\'t output anything else).\nQuestion: {question}\nthe correct answer is:'
        else:
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nthe correct answer is:'

        return prompt

def parse_generated_answer_chat_format(answer):

    if "answer is" in answer:
        idx = answer.find("answer is")
        answer = answer[idx+len("answer is"): ].strip()
        if answer.startswith(":"):
            answer = answer[1:].strip()
    return answer

def parse_gemma_mistral_answer(answer):

    candidate_answers = answer.split("\n")
    answer = ""
    i = 0
    while len(answer) < 1 and i<len(candidate_answers):
        answer = candidate_answers[i].strip()
        i += 1
    answer = parse_generated_answer_chat_format(answer)
    return answer


# ============ Llama3 Evaluation Function ============
def evaluate_with_llama3(args, model, tokenizer, data):
    """
    Evaluate retrieval results using Llama3 model

    Args:
        args: Configuration arguments
        model: Llama3 model
        tokenizer: Llama3 tokenizer
        data: Test data

    Returns:
        Dict: Evaluation metrics
    """
    em_scores_list, f1_scores_list = [], []
    processor = Llama3DataProcessor(args)
    retrieved_docs = None
    print(f"Starting evaluation on {len(data)} samples...")

    for i, example in enumerate(tqdm(data, desc="Evaluating")):
        question = example["question"]
        gold_answers = example["answers"]
        ctxs = example["ctxs"]
        if args.norag:
            prompt = processor.create_llama3_prompt(question, retrieved_docs)
        else:
            if args.prompt_based:
                retrieved_docs = retrieve_documents_by_similarity_score(question, ctxs, args)
                prompt = processor.create_llama3_prompt_score(question, retrieved_docs)
            elif args.dual_factor:
                retrieved_docs = retrieve_documents_by_similarity_andCredibility(question, ctxs, args)
                prompt = processor.create_llama3_prompt(question, retrieved_docs)
            else:
                retrieved_docs = retrieve_documents_by_similarity(question, ctxs, args)
                prompt = processor.create_llama3_prompt(question, retrieved_docs)

        # Convert chat format to model input
        input_text = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False
        )

        # Encode input
        encoded = tokenizer(
            [input_text],
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=args.answer_maxlength,
                do_sample=False,
                temperature=1.0,
                use_cache=True
            )

        # Decode generated answer
        generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:].detach().cpu()
        predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
        print("-" * 50)

        # Calculate evaluation metrics
        em_score = ems(predicted_answer, gold_answers)
        em_scores_list.append(em_score)

        if not em_score and i < 5:  # Print only the first 5 error cases
            print(f"\nError Case {i + 1}:")
            print(f"Question: {question}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Correct Answer: {gold_answers}")
            print("-" * 50)

        f1, precision, recall = f1_score(predicted_answer, gold_answers[0])
        f1_scores_list.append(f1)

    # Calculate final metrics
    metrics = {
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list)
    }

    return metrics


# ============ Gemma Evaluation Function ============
def evaluate_with_gemma(args, model, tokenizer, data):
    """
    Evaluate retrieval results using Gemma model

    Args:
        args: Configuration arguments
        model: Gemma model
        tokenizer: Gemma tokenizer
        data: Test data

    Returns:
        Dict: Evaluation metrics
    """
    em_scores_list, f1_scores_list = [], []
    processor = GemmaDataProcessor(args)
    retrieved_docs = None
    print(f"Starting evaluation on {len(data)} samples...")

    for i, example in enumerate(tqdm(data, desc="Evaluating")):
        question = example["question"]
        gold_answers = example["answers"]
        ctxs = example["ctxs"]
        if args.norag:
            messages = processor.create_gemma_prompt(question, retrieved_docs)
        else:
            if args.prompt_based:
                retrieved_docs = retrieve_documents_by_similarity_score(question, ctxs, args)
                messages = processor.create_gemma_prompt_score(question, retrieved_docs)
            else:
                retrieved_docs = retrieve_documents_by_similarity(question, ctxs, args)
                messages = processor.create_gemma_prompt(question, retrieved_docs)

        # Encode input
        encoded = tokenizer(
            messages,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=args.answer_maxlength,
                do_sample=False,
                temperature=1.0,
                use_cache=True
            )

        # Decode generated answer
        generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:].detach().cpu()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_answer = parse_gemma_mistral_answer(generated_text)

        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
        print("-" * 50)

        # Calculate evaluation metrics
        em_score = ems(predicted_answer, gold_answers)
        em_scores_list.append(em_score)

        if not em_score and i < 5:  # Print only the first 5 error cases
            print(f"\nError Case {i + 1}:")
            print(f"Question: {question}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Correct Answer: {gold_answers}")
            print("-" * 50)

        f1, precision, recall = f1_score(predicted_answer, gold_answers[0])
        f1_scores_list.append(f1)

    # Calculate final metrics
    metrics = {
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list)
    }

    return metrics

# ============ Mistral Evaluation Function ============
def evaluate_with_mistral(args, model, tokenizer, data):
    """
    Evaluate retrieval results using Mistral model

    Args:
        args: Configuration arguments
        model: Mistral model
        tokenizer: Mistral tokenizer
        data: Test data

    Returns:
        Dict: Evaluation metrics
    """
    em_scores_list, f1_scores_list = [], []
    processor = MistralDataProcessor(args)
    retrieved_docs = None
    print(f"Starting evaluation on {len(data)} samples...")

    for i, example in enumerate(tqdm(data, desc="Evaluating")):
        question = example["question"]
        gold_answers = example["answers"]
        ctxs = example["ctxs"]
        if args.norag:
            messages = processor.create_mistral_prompt(question, retrieved_docs)
        else:
            if args.prompt_based:
                retrieved_docs = retrieve_documents_by_similarity_score(question, ctxs, args)
                messages = processor.create_mistral_prompt_score(question, retrieved_docs)
            else:
                retrieved_docs = retrieve_documents_by_similarity(question, ctxs, args)
                messages = processor.create_mistral_prompt(question, retrieved_docs)

        # Encode input
        encoded = tokenizer(
            messages,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=args.answer_maxlength,
                do_sample=False,
                temperature=1.0,
                use_cache=True
            )

        # Decode generated answer
        generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:].detach().cpu()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_answer = parse_gemma_mistral_answer(generated_text)

        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
        print("-" * 50)

        # Calculate evaluation metrics
        em_score = ems(predicted_answer, gold_answers)
        em_scores_list.append(em_score)

        if not em_score and i < 5:  # Print only the first 5 error cases
            print(f"\nError Case {i + 1}:")
            print(f"Question: {question}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Correct Answer: {gold_answers}")
            print("-" * 50)

        f1, precision, recall = f1_score(predicted_answer, gold_answers[0])
        f1_scores_list.append(f1)

    # Calculate final metrics
    metrics = {
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list)
    }

    return metrics

# ============ CAG-7B Model Loader ============
def load_cag_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left"
    )
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    model.eval()
    return model, tokenizer


# ============ Data Processor ============
class CagDataProcessor:
    """Single-hop Data Processor"""

    def __init__(self, args):
        self.args = args

    def format_documents(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents, including credibility scores

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted document string
        """
        formatted_docs = []

        for i, doc in enumerate(retrieved_docs):
            score = doc["credibility"]
            text = doc["text"]
            if score <= 3:
                credibility = "Low credibility of text"
            elif score > 3 and score < 7:
                credibility = "Medium credibility of text"
            elif score >= 7:
                credibility = "High credibility of text"
            # Format document: includes credibility score and content
            formatted_doc = f"{credibility}: {text} "
            formatted_docs.append(formatted_doc)

        return "\n".join(formatted_docs)

    def create_prompt(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        Create prompt in the specified format

        Args:
            question: Question text
            retrieved_docs: List of retrieved documents

        Returns:
            str: Chat-formatted prompt
        """

        user_input = '''You are an assistant who can answer questions based on the given passages. Each passage has a credibility score that indicates the relevance and accuracy of the passage to the question. Your answer need to combine multiple passages and their credibility.Question:{question}\nDocs:{paras}\n\nYour answer should based on the given information or your internal knowledge with one or few words without the source  (just output a answer, don\'t output anything else). Answer:'''

        if retrieved_docs:
            paras = self.format_documents(retrieved_docs)
            user_input = user_input.format(question=question, paras=paras)
        else:
            user_input = f"Question: {question}\n\nAnswer (one or few words only):"

        return user_input


def parse_cag_answer(answer):
    if "Answer:" in answer:
        idx = answer.find("Answer:")
        answer = answer[idx + len("Answer:"):].strip()
        if answer.startswith(":"):
            answer = answer[1:].strip()
    return answer


# ============ Evaluation Function ============
def evaluate_with_cag(args, cag_tokenizer, cag_model, data):
    """
    Evaluate single-hop retrieval results

    Args:
        args: Configuration arguments
        cag_tokenizer: CAG model tokenizer
        cag_model: CAG model
        data: Test data

    Returns:
        Dict: Evaluation metrics
    """
    em_scores_list, f1_scores_list, accuracy_list = [], [], []
    processor = CagDataProcessor(args)

    print(f"Starting evaluation on {len(data)} samples...")

    for i, example in enumerate(tqdm(data, desc="Evaluating")):
        question = example["question"]
        gold_answers = example["answers"]
        ctxs = example["ctxs"]

        # Execute single-hop retrieval
        retrieved_doc = retrieve_documents_by_similarity_score(question, ctxs, args)

        # Create prompt
        prompt = processor.create_prompt(question, retrieved_doc)

        # Encode input
        encoded = cag_tokenizer(
            prompt,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
        # Generate answer
        with torch.no_grad():
            outputs = cag_model.generate(
                **model_inputs,
                max_new_tokens=args.answer_maxlength,
                do_sample=False,
                temperature=1.0
            )

        # Decode generated answer
        generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:].detach().cpu()
        generated_text = cag_tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_answer = generated_text.strip()
        predicted_answer = parse_cag_answer(predicted_answer)
        print(predicted_answer)
        # Calculate evaluation metrics
        acc = accuracy(predicted_answer, gold_answers)
        accuracy_list.append(acc)
        # if not acc and i < 5:  # Print only the first 5 cases
        #     print(f"\nError Case {i + 1}:")
        #     print(f"Question: {question}")
        #     print(f"Predicted Answer: {predicted_answer}")
        #     print(f"Correct Answer: {gold_answers}")
        #     print("-" * 50)
        em_score = ems(predicted_answer, gold_answers)
        em_scores_list.append(em_score)
        if not em_score and i < 5:  # Print only the first 5 error cases
            print(f"\nError Case {i + 1}:")
            print(f"Question: {question}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Correct Answer: {gold_answers}")
            print("-" * 50)

        f1, precision, recall = f1_score(predicted_answer, gold_answers[0])
        f1_scores_list.append(f1)

    metrics = {
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list),
        "accuracy": np.mean(accuracy_list),
    }

    return metrics


# ============ Knowledge-R1 Model Loader ============
def load_KnowledgeR1_model_tokenizer(model_path):
    """
    Load Knowledge-R1 model and tokenizer

    Args:
        model_path: Model path

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading Knowledge-R1 model: {model_path}")

    # Load tokenizer - Use use_fast=False to avoid fast tokenizer compatibility issues
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True,
            use_fast=False  # Use slow tokenizer
        )
    except Exception as e:
        print(f"Slow tokenizer load failed, trying alternatives: {e}")
        # Fallback: Specify tokenizer class directly
        from transformers import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True
        )

    # Set pad_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Setting padding token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.to(device)

    model.eval()
    print("Knowledge-R1 model loaded successfully!")
    return model, tokenizer


# ============ Knowledge-R1 Data Processor ============
class KnowledgeR1DataProcessor:
    """Knowledge-R1 Data Processor"""

    def __init__(self, args):
        self.args = args
        self.system_prompt = getattr(args, 'system_prompt',
                                     "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
                                     "The reasoning process MUST BE enclosed within <think> </think> tags. "
                                     "The final answer MUST BE put in \\boxed{}.")

    def get_contexts(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted context string
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            contexts.append(f"Passage-{i}: {text}")
        return "\n".join(contexts)

    def create_knowledge_r1_prompt(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        Create a prompt formatted for Knowledge-R1

        Args:
            question: Question text
            retrieved_docs: List of retrieved documents

        Returns:
            str: Formatted prompt string
        """
        if retrieved_docs:
            contexts = self.get_contexts(retrieved_docs)
            user_content = (
                f"Given the following information:  \n{contexts}\n"
                f"please only output the answer to the question.\n"
                f"Question:{question}\n"
                f"the correct answer is:"
            )
        else:
            user_content = (
                f"Answer the following question based on your internal knowledge with one or few words.\n"
                f"Question:  {question}\n"
                f"the correct answer is:"
            )

        # Build complete prompt (using Qwen chat format)
        prompt_text = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return prompt_text


# ============ Knowledge-R1 Answer Parser Functions ============
def extract_boxed_content(text: str) -> str:
    """
    Extract content inside \\boxed{} from the text

    Args:
        text: Complete generated output from the model

    Returns:
        str: Extracted answer, returns None if not found
    """
    # Match \boxed{... } format
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, text)

    if matches:
        # Return the last matched answer
        return matches[-1].strip()

    return None


def parse_knowledge_r1_answer(answer: str) -> str:
    """
    Parse the output of Knowledge-R1 model to extract the answer within \\boxed{}

    Args:
        answer: Complete generated output from the model

    Returns:
        str: Extracted answer
    """
    # Try to extract answer from \boxed{}
    boxed_answer = extract_boxed_content(answer)
    if boxed_answer:
        return boxed_answer

    # If \boxed{} is not found, try alternative extraction methods
    # Check if </think> tag exists, extract content after it
    if '</think>' in answer:
        after_think = answer.split('</think>')[-1].strip()
        # Try extracting \boxed{} from the remaining content again
        boxed_answer = extract_boxed_content(after_think)
        if boxed_answer:
            return boxed_answer
        # If still not found, return the first non-empty line after </think>
        lines = [l.strip() for l in after_think.split('\n') if l.strip()]
        if lines:
            # Clean possible format markers
            first_line = lines[0]
            # Remove possible prefixes like "the correct answer is:"
            if "answer is" in first_line.lower():
                idx = first_line.lower().find("answer is")
                first_line = first_line[idx + len("answer is"):].strip()
                if first_line.startswith(": "):
                    first_line = first_line[1:].strip()
            return first_line

    # Final fallback: return the last non-empty line
    lines = [l.strip() for l in answer.split('\n') if l.strip()]
    if lines:
        return lines[-1]

    return answer.strip()


def extract_thinking_and_answer(response: str) -> Dict:
    """
    Extract thinking process and final answer from Knowledge-R1 response

    Args:
        response: Complete model response

    Returns:
        Dict: Dictionary containing 'thinking' and 'answer'
    """
    thinking = ""
    answer = ""

    # Extract content within <think> </think>
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, response, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()

    # Extract answer
    answer = parse_knowledge_r1_answer(response)

    # Extract content within <answer> </answer> again (removing tags)
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_match = re.search(answer_pattern, answer, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    return {
        "thinking": thinking,
        "answer": answer,
        "raw_response": response
    }


# ============ Knowledge-R1 Evaluation Function ============
def evaluate_with_knowledge_r1(args, model, tokenizer, data):
    """
    Evaluate retrieval results using Knowledge-R1 model

    Args:
        args: Configuration arguments
        model: Knowledge-R1 model
        tokenizer: Knowledge-R1 tokenizer
        data: Test data

    Returns:
        Dict: Evaluation metrics
    """
    em_scores_list, f1_scores_list, accuracy_list = [], [], []
    processor = KnowledgeR1DataProcessor(args)

    # Get generation parameters
    max_new_tokens = getattr(args, 'max_tokens', getattr(args, 'answer_maxlength', 2048))
    temperature = getattr(args, 'temperature', 0.0)
    top_p = getattr(args, 'top_p', 0.95)

    print(f"Starting evaluation on {len(data)} samples...")
    print(f"Generation parameters: max_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")

    for i, example in enumerate(tqdm(data, desc="Evaluating")):
        question = example["question"]
        gold_answers = example["answers"]
        ctxs = example.get("ctxs", [])

        # Choose whether to use RAG based on configuration
        if getattr(args, 'norag', False) or not ctxs:
            retrieved_docs = None
        else:
            # Use similarity-based retrieval (ignores credibility)
            retrieved_docs = retrieve_documents_by_similarity(question, ctxs, args)

        # Create prompt
        prompt = processor.create_knowledge_r1_prompt(question, retrieved_docs)

        # Encode input
        encoded = tokenizer(
            [prompt],
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=getattr(args, 'max_model_len', 8192)
        ).to(device)

        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

        # Generate answer
        with torch.no_grad():
            if temperature > 0:
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=getattr(args, 'repetition_penalty', 1.0),
                    use_cache=True
                )
            else:
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True
                )

        # Decode generated answer
        generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:].detach().cpu()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse answer (extract thinking and answer)
        parsed_result = extract_thinking_and_answer(generated_text)
        predicted_answer = parsed_result["answer"]


        # Print partial results for debugging
        if i < 3:
            print(f"\nSample {i + 1}:")
            print(f"Question: {question}")
            if parsed_result['thinking']:
                thinking_preview = parsed_result['thinking'][: 200] + "..." if len(parsed_result['thinking']) > 200 else \
                parsed_result['thinking']
                print(f"Thinking Process: {thinking_preview}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Correct Answer: {gold_answers}")
            print("-" * 50)

        # Calculate evaluation metrics
        em_score = ems(predicted_answer, gold_answers)
        em_scores_list.append(em_score)

        acc = accuracy(predicted_answer, gold_answers)
        accuracy_list.append(acc)

        if not em_score and i < 5:  # Print only the first 5 error cases
            print(f"\nError Case {i + 1}:")
            print(f"Question: {question}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Correct Answer: {gold_answers}")
            print("-" * 50)

        f1, precision, recall = f1_score(predicted_answer, gold_answers[0])
        f1_scores_list.append(f1)

    # Calculate final metrics
    metrics = {
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list),
        "accuracy": np.mean(accuracy_list),
    }

    return metrics

# ============ Parameter Setup ============
def setup_parser():
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_data_file", type=str, default="data/hotpotqa/dev_with_kgs.json",
                        help="Input data file path")
    parser.add_argument("--model_type", type=str, choices=["cag", "llama3", "gemma","mistral","knowledger1"], default="llama3",
                        help="Model type: cag for CAG-7B, llama3 for Llama3-8B-Instruct, gemma for Gemma-7B, mistral for Mistral-7B, knowledger1 for Knowledge-R1")
    parser.add_argument("--model_path", type=str, required=True, help="CAG-7B model path")
    parser.add_argument("--context_nums", type=int, default=5, help="Number of retrieved documents")
    parser.add_argument("--answer_maxlength", type=int, default=25, help="Maximum length of the answer")
    parser.add_argument("--fake_num", type=int, default=1)
    parser.add_argument("--prompt_based", action="store_true",
                        help="Run prompt based")
    parser.add_argument("--dual_factor", action="store_true",
                        help="Run prompt based")
    parser.add_argument("--norag", action="store_true",
                        help="Run inference without context")
    parser.add_argument("--exclusion", action="store_true",
                        help="This option will filter documents with credibility scores below the threshold and perform inference.")

    # Qwen/Knowledge-R1 specific arguments
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate (for Qwen)")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Maximum model context length")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty")
    parser.add_argument("--system_prompt", type=str,
                        default="""You are a helpful assistant. After the user asks a question, you first think carefully and then give the answer.
When responding, please keep the following points in mind: 
 - The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
 - Output your final answer directly between the tag <answer> </answer> without any intermediate steps.
Here is an exmaple: 
Question: 
what is the capital of China?
<think> reasoning process here </think>
<answer> BeiJing </answer>""",
                        help="System prompt for the model (for Qwen)")

    args = parser.parse_args()
    return args


# ============ Main Function ============
def main():
    """Main Function"""
    # Set up parameters
    args = setup_parser()

    print("=" * 80)
    print("=" * 80)
    print(f"Number of retrieved documents: {args.context_nums}")
    print(f"Model path: {args.model_path}")
    print("=" * 80)

    # Load test data
    print("Step 1: Loading test data...")
    data = load_json(args.input_data_file)
    print(f"Dataset size: {len(data)} samples")
    if args.model_type == "llama3":
        model, tokenizer = load_llama3_model_tokenizer(args.model_path)
        print("Step 3: Starting evaluation (using Llama3)...")
        metrics = evaluate_with_llama3(args, model, tokenizer, data)
    elif args.model_type == "gemma":
        model, tokenizer = load_gemma_model_tokenizer(args.model_path)
        print("Step 3: Starting evaluation (using Gemma-7B)...")
        metrics = evaluate_with_gemma(args, model, tokenizer, data)
    elif args.model_type == "mistral":
        model, tokenizer = load_mistral_model_tokenizer(args.model_path)
        print("Step 3: Starting evaluation (using Mistral-7B)...")
        metrics = evaluate_with_mistral(args, model, tokenizer, data)
    elif args.model_type == "knowledger1":
        # Knowledge-R1 model
        model, tokenizer = load_KnowledgeR1_model_tokenizer(args.model_path)
        print("Step 2: Starting evaluation (using Knowledge-R1)...")
        metrics = evaluate_with_knowledge_r1(args, model, tokenizer, data)
    else:
        # Load CAG-7B model
        print("Step 2: Loading CAG-7B model...")
        cag_model, cag_tokenizer = load_cag_model_tokenizer(args)
        # Execute evaluation
        print("Step 3: Starting retrieval and evaluation...")
        metrics = evaluate_with_cag(args, cag_tokenizer, cag_model, data)

    # Output results
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
