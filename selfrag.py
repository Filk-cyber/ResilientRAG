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
from readers.metrics import ems, f1_score,accuracy
import re
from concurrent.futures import ThreadPoolExecutor

# Global variables
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============ Self-RAG Configuration ============


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Self-RAG Prompt Templates
SELFRAG_PROMPT_DICT = {
    "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:\n",
# Added: Concise output template
    "prompt_concise": "### Instruction:\nGiven the following question, please only output the answer with one or few words.\nQuestion: {instruction}\nThe correct answer is:\n\n### Response:\n",
}

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


# ============ Self-RAG Special Token Loading ============
def load_selfrag_special_tokens(tokenizer, use_grounding=True, use_utility=False):
    """
    Load special token IDs required for Self-RAG

    Args:
        tokenizer: tokenizer object
        use_grounding: Whether to use grounding tokens
        use_utility: Whether to use utility tokens

    Returns:
        tuple: (ret_tokens, rel_tokens, grd_tokens, ut_tokens)
    """
    ret_tokens = {
        "[Retrieval]": tokenizer.convert_tokens_to_ids("[Retrieval]"),
        "[No Retrieval]": tokenizer.convert_tokens_to_ids("[No Retrieval]"),
        "[Continue to Use Evidence]": tokenizer.convert_tokens_to_ids("[Continue to Use Evidence]"),
    }

    rel_tokens = {
        "[Relevant]": tokenizer.convert_tokens_to_ids("[Relevant]"),
        "[Irrelevant]": tokenizer.convert_tokens_to_ids("[Irrelevant]"),
    }

    grd_tokens = None
    if use_grounding:
        grd_tokens = {
            "[Fully supported]": tokenizer.convert_tokens_to_ids("[Fully supported]"),
            "[Partially supported]": tokenizer.convert_tokens_to_ids("[Partially supported]"),
            "[No support / Contradictory]": tokenizer.convert_tokens_to_ids("[No support / Contradictory]"),
        }

    ut_tokens = None
    if use_utility:
        ut_tokens = {
            "[Utility: 1]": tokenizer.convert_tokens_to_ids("[Utility:1]"),
            "[Utility:2]": tokenizer.convert_tokens_to_ids("[Utility:2]"),
            "[Utility:3]": tokenizer.convert_tokens_to_ids("[Utility:3]"),
            "[Utility:4]": tokenizer.convert_tokens_to_ids("[Utility:4]"),
            "[Utility: 5]": tokenizer.convert_tokens_to_ids("[Utility:5]"),
        }

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


# ============ Self-RAG Model Loading (using transformers) ============
def load_selfrag_model_tokenizer(model_path):
    """
    Load Self-RAG model and tokenizer (using transformers)

    Args:
        model_path: Model path

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading Self-RAG model:  {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left"
    )

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Setting padding token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
    print("Self-RAG model loaded successfully!")
    return model, tokenizer


# ============ Self-RAG Data Processor ============
class SelfRAGDataProcessor:
    """Self-RAG Data Processor"""

    def __init__(self, args):
        self.args = args

    def format_paragraph(self, doc: Dict) -> Dict:
        """
        Format documents into the paragraph format required by Self-RAG
        """
        text = doc.get("text", "")
        return {"text": text}

    def create_selfrag_prompt(self, question: str, concise: bool = True) -> str:
        """
        Create a prompt formatted for Self-RAG

        Args:
            question: Question text
            concise: Whether to use the concise output template

        Returns:
            str: Formatted prompt
        """
        if concise:
            return SELFRAG_PROMPT_DICT["prompt_concise"].format(instruction=question)
        else:
            return SELFRAG_PROMPT_DICT["prompt_no_input"].format(instruction=question)


# ============ Self-RAG Generation Function (using transformers) ============
def selfrag_generate_with_scores(model, tokenizer, prompts: List[str], max_new_tokens: int):
    """
    Generate text using transformers and obtain logprobs

    Args:
        model: transformers model
        tokenizer: tokenizer
        prompts: List of prompts
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        List[Dict]: Generation results for each prompt, including token_ids, text, logprobs, seq_score
    """
    results = []

    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        input_length = encoded["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )

        generated_ids = outputs.sequences[0, input_length:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Calculate logprobs for each position
        logprobs_list = []
        cumulative_logprob = 0.0

        if hasattr(outputs, 'scores') and outputs.scores:
            for step_idx, scores in enumerate(outputs.scores):
                # scores shape: (batch_size, vocab_size)
                log_probs = torch.nn.functional.log_softmax(scores[0], dim=-1)

                # Get log prob of the current token
                if step_idx < len(generated_ids):
                    token_id = generated_ids[step_idx]
                    token_logprob = log_probs[token_id].item()
                    cumulative_logprob += token_logprob

                # Store logprobs for all tokens (used to calculate probabilities of special tokens)
                logprobs_dict = {}
                for tid in range(log_probs.shape[0]):
                    logprobs_dict[tid] = log_probs[tid].item()
                logprobs_list.append(logprobs_dict)

        seq_score = cumulative_logprob / max(len(generated_ids), 1)

        results.append({
            "token_ids": generated_ids,
            "text": generated_text,
            "logprobs": logprobs_list,
            "seq_score": seq_score,
            "cumulative_logprob": cumulative_logprob
        })

    return results


# ============ Self-RAG Single-step Generation Batch Processing ============
def selfrag_run_step_generation_batch(model, tokenizer, prompt, paragraphs, max_new_tokens,
                                      rel_tokens=None, grd_tokens=None, ret_tokens=None, ut_tokens=None,
                                      threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False):
    """
    Self-RAG single-step batch generation (using transformers)
    """
    if paragraphs is not None:
        aug_prompts = [
            prompt + "[Retrieval]" + "<paragraph>{}</paragraph>".format(
                paragraph["text"]
            ) for paragraph in paragraphs
        ]
    else:
        aug_prompts = [prompt]

    # Generate using transformers
    preds = selfrag_generate_with_scores(model, tokenizer, aug_prompts, max_new_tokens)

    relevance_score_dict = {}
    grd_score_dict = {}
    ut_score_dict = {}
    overall_scores = {}
    final_preds = []

    for p_idx, pred in enumerate(preds):
        pred_token_ids = pred["token_ids"]
        pred_text = pred["text"]
        pred_log_probs = pred["logprobs"]
        seq_score = pred["seq_score"]

        relevance_score_dict.setdefault(p_idx, {})
        grd_score_dict.setdefault(p_idx, {})
        ut_score_dict.setdefault(p_idx, {})

        # Calculate relevance score (at the first generated token position)
        if len(pred_log_probs) > 0:
            for tok, tok_id in rel_tokens.items():
                if tok_id in pred_log_probs[0]:
                    prob = np.exp(pred_log_probs[0][tok_id])
                else:
                    prob = 1e-10
                relevance_score_dict[p_idx][tok] = prob

        # Calculate grounding score
        if grd_tokens is not None:
            groundness_token_appear_indices = []
            for tok_idx, tok in enumerate(pred_token_ids):
                if tok in list(grd_tokens.values()):
                    groundness_token_appear_indices.append(tok_idx)
                    break

            if len(groundness_token_appear_indices) > 0:
                idx = groundness_token_appear_indices[0]
                if idx < len(pred_log_probs):
                    for token, token_id in grd_tokens.items():
                        if token_id in pred_log_probs[idx]:
                            prob = np.exp(pred_log_probs[idx][token_id])
                        else:
                            prob = 1e-10
                        grd_score_dict[p_idx][token] = prob

        # Calculate utility score
        if ut_tokens is not None:
            utility_token_appear_indices = []
            for tok_idx, tok in enumerate(pred_token_ids):
                if tok in list(ut_tokens.values()):
                    utility_token_appear_indices.append(tok_idx)

            if len(utility_token_appear_indices) > 0:
                idx = utility_token_appear_indices[0]
                if idx < len(pred_log_probs):
                    for token, token_id in ut_tokens.items():
                        if token_id in pred_log_probs[idx]:
                            prob = np.exp(pred_log_probs[idx][token_id])
                        else:
                            prob = 1e-10
                        ut_score_dict[p_idx][token] = prob

        # Calculate final score
        rel_sum = np.sum(list(relevance_score_dict[p_idx].values())) if relevance_score_dict[p_idx] else 1
        relevance_score = relevance_score_dict[p_idx].get("[Relevant]", 0) / rel_sum if rel_sum > 0 else 0

        if len(grd_score_dict[p_idx]) == 3:
            gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
            if gt_sum > 0:
                ground_score = (grd_score_dict[p_idx].get("[Fully supported]", 0) / gt_sum) + \
                               0.5 * (grd_score_dict[p_idx].get("[Partially supported]", 0) / gt_sum)
            else:
                ground_score = 0.0
        else:
            ground_score = 0.0

        if len(ut_score_dict[p_idx]) == 5:
            ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
            if ut_sum > 0:
                ut_scores_weights = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum([
                    ut_scores_weights[i] * (ut_score_dict[p_idx].get("[Utility:{}]".format(i + 1), 0) / ut_sum)
                    for i in range(5)
                ])
            else:
                utility_score = 0.0
        else:
            utility_score = 0.0

        if use_seqscore:
            final_score = np.exp(seq_score) + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score
        else:
            final_score = w_rel * relevance_score + w_sup * ground_score + w_use * utility_score

        overall_scores[p_idx] = {
            "final_score": final_score,
            "relevance_score": relevance_score,
            "ground_score": ground_score,
            "utility_score": utility_score,
        }

        # Process [No Retrieval] token to determine if further retrieval is needed
        if "[No Retrieval]" in pred_text and ret_tokens is not None:
            ret_token_appear_indices = []
            substrings = pred_text.split("[No Retrieval]")

            for tok_idx, tok in enumerate(pred_token_ids):
                if tok == ret_tokens["[No Retrieval]"]:
                    ret_token_appear_indices.append(tok_idx)

            retrieval_remap = {}
            for order, idx in enumerate(ret_token_appear_indices):
                if idx < len(pred_log_probs):
                    ret_token_score_dict = {}
                    for tok, tok_id in ret_tokens.items():
                        if tok_id in pred_log_probs[idx]:
                            ret_token_score_dict[tok] = np.exp(pred_log_probs[idx][tok_id])
                        else:
                            ret_token_score_dict[tok] = 1e-10

                    denom = ret_token_score_dict.get("[Retrieval]", 0) + ret_token_score_dict.get("[No Retrieval]", 0)
                    if denom > 0 and threshold is not None:
                        retrieve_prob = (ret_token_score_dict.get("[Retrieval]", 0) +
                                         ret_token_score_dict.get("[Continue to Use Evidence]", 0)) / denom
                        retrieval_remap[order] = retrieve_prob > threshold
                    else:
                        retrieval_remap[order] = False
                else:
                    retrieval_remap[order] = False

            processed_pred = ""
            for substr_i, substring in enumerate(substrings):
                if substr_i in retrieval_remap and retrieval_remap[substr_i]:
                    processed_pred += substring + "[Retrieval]"
                else:
                    processed_pred += substring + "[No Retrieval]"
            pred_text = processed_pred

        final_preds.append(pred_text)

    scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores]
    return final_preds, scores, overall_scores


# ============ Self-RAG Beam Search Inference (using transformers) ============
def selfrag_call_model_beam_batch(prompt, model, tokenizer, max_new_tokens, ctxs, query,
                                  rel_tokens, grd_tokens, ret_tokens, ut_tokens,
                                  threshold, beam_width, max_depth,
                                  w_rel=1.0, w_sup=1.0, w_use=0.5,
                                  use_seqscore=False, ignore_cont=False,
                                  mode="adaptive_retrieval"):
    """
    Main function for Self-RAG Beam Search inference (using transformers)
    """
    prediction_tree = {}

    # No retrieval mode
    if mode == "no_retrieval":
        prompt_with_tag = prompt + "[No Retrieval]"
        preds = selfrag_generate_with_scores(model, tokenizer, [prompt_with_tag], max_new_tokens)
        pred_text = preds[0]["text"].split("\n\n")[0]
        return pred_text, prediction_tree

    do_retrieve = False

    # Always retrieve mode
    if mode == "always_retrieve":
        do_retrieve = True
    else:
        # Adaptive retrieval mode: Generate a short segment first to see if retrieval is needed
        preds = selfrag_generate_with_scores(model, tokenizer, [prompt], max_new_tokens=25)
        pred_text = preds[0]["text"].split("\n\n")[0]
        pred_log_probs = preds[0]["logprobs"]

        if "[Retrieval]" not in pred_text:
            do_retrieve = False
        else:
            if threshold is None:
                do_retrieve = False
            elif len(pred_log_probs) > 0:
                ret_token_score_dict = {}
                for tok, tok_id in ret_tokens.items():
                    if tok_id in pred_log_probs[0]:
                        ret_token_score_dict[tok] = np.exp(pred_log_probs[0][tok_id])
                    else:
                        ret_token_score_dict[tok] = 1e-10

                denom = ret_token_score_dict.get("[Retrieval]", 0) + ret_token_score_dict.get("[No Retrieval]", 0)
                if denom > 0:
                    retrieve_prob = ret_token_score_dict.get("[Retrieval]", 0) / denom
                    do_retrieve = retrieve_prob > threshold
                else:
                    do_retrieve = False
            else:
                do_retrieve = False

    # No retrieval needed
    if not do_retrieve:
        prompt_with_tag = prompt + "[No Retrieval]"
        preds = selfrag_generate_with_scores(model, tokenizer, [prompt_with_tag], max_new_tokens)
        pred_text = preds[0]["text"].split("\n\n")[0]
        return pred_text, prediction_tree

    # Retrieval needed: Execute beam search
    curr_depth = 1
    terminated = False
    node_id = 0
    levels = {}

    prediction_tree[node_id] = {
        "prompt": prompt,
        "pred": "[Retrieval]",
        "processed_pred": "",
        "score": None,
        "ctx": None,
        "parent": None
    }
    levels[0] = [0]

    while curr_depth < max_depth:
        levels[curr_depth] = []

        if curr_depth - 1 in levels and not terminated:
            for node in levels[curr_depth - 1]:
                pred = prediction_tree[node]["pred"]
                if pred == "</s>":
                    terminated = True
                    continue

                node_prompt = prediction_tree[node]["prompt"]
                prev_generation = prediction_tree[node]["processed_pred"]
                score = prediction_tree[node]["score"]

                if "[Retrieval]" in pred:
                    preds, scores, overall_score_dict = selfrag_run_step_generation_batch(
                        model, tokenizer, node_prompt + prev_generation, ctxs, max_new_tokens,
                        rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                        threshold=threshold, w_rel=w_rel, w_sup=w_sup, w_use=w_use, use_seqscore=use_seqscore
                    )

                    for i, (pred_text, p_score) in enumerate(zip(preds, scores)):
                        node_id += 1
                        node_score = p_score * score if score is not None else p_score

                        prediction_tree[node_id] = {
                            "prompt": node_prompt + prev_generation,
                            "pred": pred_text,
                            "score": node_score,
                            "ctx": ctxs[i] if i < len(ctxs) else None,
                            "parent": node,
                            "overall_score_dict": overall_score_dict
                        }

                        if "[Retrieval]" in pred_text:
                            gen_result_index = pred_text.index("[Retrieval]")
                            prev_gen = pred_text[:gen_result_index]
                        else:
                            prev_gen = pred_text

                        prediction_tree[node_id]["processed_pred"] = prev_gen
                        levels[curr_depth].append(node_id)

            # Select top-k nodes
            if levels[curr_depth]:
                node2score = {nid: prediction_tree[nid]["score"] for nid in levels[curr_depth]}
                top_nodes = sorted(
                    node2score.items(),
                    key=lambda x: x[1] if x[1] is not None else 0,
                    reverse=True
                )[:beam_width]
                levels[curr_depth] = [node[0] for node in top_nodes]

            curr_depth += 1
        else:
            break

    # Build final result
    levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0}

    if not levels:
        return "", prediction_tree

    best_selections = {}
    final_level = max(levels.keys())

    for path_i, node in enumerate(levels[final_level]):
        if node == 0:
            break
        best_selections[path_i] = [node]
        current_node = node

        while current_node is not None:
            parent = prediction_tree[current_node]["parent"]
            if parent is not None:
                best_selections[path_i] = [parent] + best_selections[path_i]
            current_node = parent

    final_prediction = {}
    splitted_sentences = {}
    original_splitted_sentences = {}
    result_ctxs = {}

    for path_i, nodes in best_selections.items():
        final_prediction[path_i] = " ".join([
            prediction_tree[node]["processed_pred"]
            for node in nodes
            if node is not None and (
                    not ignore_cont or "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
            )
        ])

        splitted_sentences[path_i] = [
            prediction_tree[node]["processed_pred"]
            for node in nodes
            if node is not None and (
                    not ignore_cont or "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
            )
        ]

        original_splitted_sentences[path_i] = [
            prediction_tree[node]["pred"]
            for node in nodes
            if node is not None and (
                    not ignore_cont or "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
            )
        ]

        result_ctxs[path_i] = [
            prediction_tree[node]["ctx"]
            for node in nodes
            if node is not None and (
                    not ignore_cont or "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
            )
        ]

    result = {
        "final_prediction": final_prediction,
        "splitted_sentences": splitted_sentences,
        "original_splitted_sentences": original_splitted_sentences,
        "best_selections": best_selections,
        "ctxs": result_ctxs,
        "prediction_tree": prediction_tree
    }

    return final_prediction, result


# ============ Self-RAG Post-processing Functions ============
def selfrag_postprocess(pred):
    """Post-process text generated by Self-RAG"""
    special_tokens = [
        "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
        "[No Retrieval]", "[Retrieval]", "[Irrelevant]", "[Relevant]",
        "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]",
        "[Utility:3]", "[Utility:4]", "[Utility:5]",
        "[Continue to Use Evidence]"
    ]

    for token in special_tokens:
        pred = pred.replace(token, "")

    if "</s>" in pred:
        pred = pred.replace("</s>", "")

    return pred.strip()


def selfrag_fix_spacing(text):
    """Fix spacing issues in text"""
    return " ".join(text.split())


def selfrag_extract_answer(pred):
    """
    Extract the final answer from the generated text
    Handles various possible output formats
    """
    pred = selfrag_postprocess(pred)

    # Convert to lowercase for matching
    pred_lower = pred.lower()

    # Handle "The answer is: XXX" or "The correct answer is: XXX" formats
    patterns = [
        "the correct answer is:",
        "the correct answer is",
        "correct answer is:",
        "correct answer is",
        "the answer is:",
        "the answer is",
        "answer is:",
        "answer is",
        "answer:",
    ]

    for pattern in patterns:
        if pattern in pred_lower:
            # Find the position of the pattern
            idx = pred_lower.find(pattern)
            # Extract the content after the pattern
            result = pred[idx + len(pattern):].strip()
            if result:
                return result

    return pred

# ============ Self-RAG Evaluation Function ============
def evaluate_with_selfrag(args, model, tokenizer, data):
    """
    Evaluate using Self-RAG model (using transformers)

    Args:
        args: Configuration arguments
        model: Self-RAG model
        tokenizer: tokenizer
        data: Test data

    Returns:
        Dict: Evaluation metrics
    """
    em_scores_list, f1_scores_list,accuracy_list = [], [],[]
    processor = SelfRAGDataProcessor(args)

    # Load special tokens
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_selfrag_special_tokens(
        tokenizer,
        use_grounding=args.use_grounding,
        use_utility=args.use_utility
    )

    print(f"Starting Self-RAG evaluation on {len(data)} samples...")

    for i, example in enumerate(tqdm(data, desc="Self-RAG Evaluating")):
        question = example["question"]
        gold_answers = example["answers"]
        ctxs = example["ctxs"]

        # Use the existing retrieval function to get documents
        retrieved_docs = retrieve_documents_by_similarity(question, ctxs, args)

        # Convert to Self-RAG format
        selfrag_ctxs = [processor.format_paragraph(doc) for doc in retrieved_docs]

        # Create prompt
        prompt = processor.create_selfrag_prompt(question)

        # Call Self-RAG beam search inference
        final_pred, intermediate = selfrag_call_model_beam_batch(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            ctxs=selfrag_ctxs,
            query=question,
            rel_tokens=rel_tokens,
            grd_tokens=grd_tokens,
            ret_tokens=ret_tokens,
            ut_tokens=ut_tokens,
            threshold=args.threshold,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            w_rel=args.w_rel,
            w_sup=args.w_sup,
            w_use=args.w_use,
            use_seqscore=args.use_seqscore,
            ignore_cont=args.ignore_cont,
            mode=args.selfrag_mode
        )

        # Extract final prediction result
        if isinstance(final_pred, dict) and 0 in final_pred:
            predicted_answer = selfrag_fix_spacing(selfrag_extract_answer(final_pred[0]))
        elif isinstance(final_pred, str):
            predicted_answer = selfrag_fix_spacing(selfrag_extract_answer(final_pred))
        else:
            predicted_answer = ""

        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
        print("-" * 50)

        # Calculate evaluation metrics
        em_score = ems(predicted_answer, gold_answers)
        em_scores_list.append(em_score)

        acc = accuracy(predicted_answer, gold_answers)
        accuracy_list.append(acc)

        if not em_score and i < 5:
            print(f"\nError Case {i + 1}:")
            print(f"Question: {question}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Correct Answer: {gold_answers}")
            print("-" * 50)

        f1, _, _ = f1_score(predicted_answer, gold_answers[0])
        f1_scores_list.append(f1)

    metrics = {
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list),
        "accuracy": np.mean(accuracy_list),
    }

    return metrics


# ============ Modify setup_parser to add Self-RAG parameters ============
def setup_parser():
    """Set up command line arguments (including Self-RAG)"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Original parameters
    parser.add_argument("--input_data_file", type=str, default="data/hotpotqa/dev_with_kgs.json",
                        help="Input data file path")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--context_nums", type=int, default=5, help="Number of retrieved documents")
    parser.add_argument("--fake_num", type=int, default=1)
    # Self-RAG specific parameters
    parser.add_argument("--max_new_tokens", type=int, default=25,
                        help="Self-RAG maximum generated tokens")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Self-RAG retrieval threshold")
    parser.add_argument("--beam_width", type=int, default=2,
                        help="Self-RAG beam width")
    parser.add_argument("--max_depth", type=int, default=2,
                        help="Self-RAG max depth")
    parser.add_argument("--w_rel", type=float, default=1.0,
                        help="Document relevance weight")
    parser.add_argument("--w_sup", type=float, default=1.0,
                        help="Support weight")
    parser.add_argument("--w_use", type=float, default=1.0,
                        help="Utility weight")
    parser.add_argument("--use_grounding", action="store_true",
                        help="Use grounding scores")
    parser.add_argument("--use_utility", action="store_true",
                        help="Use utility scores")
    parser.add_argument("--use_seqscore", action="store_true",
                        help="Use sequence scores")
    parser.add_argument("--ignore_cont", action="store_true",
                        help="Ignore contradictory content")
    parser.add_argument("--selfrag_mode", type=str, default="always_retrieve",
                        choices=["adaptive_retrieval", "no_retrieval", "always_retrieve"],
                        help="Self-RAG inference mode")

    args = parser.parse_args()
    return args


# ============ Modified main function ============
def main():
    """Main function (supports Self-RAG)"""
    args = setup_parser()

    print("=" * 80)
    print(f"Number of retrieved documents: {args.context_nums}")
    print(f"Model path: {args.model_path}")
    print("=" * 80)

    # Load test data
    print("Step 1: Loading test data...")
    data = load_json(args.input_data_file)
    print(f"Dataset size: {len(data)} samples")

    # Self-RAG model
    print("Step 2: Loading Self-RAG model...")
    model, tokenizer = load_selfrag_model_tokenizer(args.model_path)
    print("Step 3: Starting evaluation (using Self-RAG)...")
    metrics = evaluate_with_selfrag(args, model, tokenizer, data)


    # Output results
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
