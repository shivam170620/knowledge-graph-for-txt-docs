"""
RAGAS Evaluation Pipeline using Google Gemini LLM and SentenceTransformer embeddings.

This script evaluates RAG (Retrieval-Augmented Generation) outputs using standard RAGAS metrics:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
- Answer Similarity ( this one not ran due to some issues debugging it)

Key Components:
- ✅ GeminiLLMWrapper: Wraps Google Gemini API into a RAGAS-compatible LLM interface.
- ✅ HuggingfaceEmbeddings: Provides vector embeddings for context-grounded evaluation.
- ✅ EvaluationDataset: RAGAS format for each QA pair including query, context, answer, and ground truth.
- ✅ Modular metric loop: Evaluates each sample sequentially, respecting rate limits.

Why this is useful:
- Enables fine-grained per-question scoring of LLM responses.
- Supports Open-Source + Proprietary LLMs via pluggable wrappers.
- Runs safely with token rate limits via sequential execution and delays.

Challenges Addressed:
- ✳️ Gemini does not have native RAGAS compatibility — solved via `GeminiClient` + `GeminiLLMWrapper`.
- ✳️ Input formats vary — handled with `process_item()` abstraction.
- ✳️ Preventing rate-limit violations — achieved via delay and graceful retries.

----------------------------------------------------------------------
⚠️ Execution Throttling & Delay Justification:

Although Gemini's free tier supports ~15 requests per minute and ~1 million tokens per minute (TPM),
we **intentionally add delays (e.g., time.sleep(5–10 seconds))** between each evaluation step for the following reasons:

1. **Avoid burst throttling:** Gemini may throttle burst requests even within stated limits.
2. **Ensure reliability:** Delays help prevent API 429 rate-limit errors during long runs (50+ samples).
3. **Multiple LLM calls per sample:** RAGAS evaluates multiple metrics per sample — each may call the LLM once.
4. **Consistent generation quality:** Sequential, paced LLM requests reduce noise from system load.
5. **Friendly to free-tier quotas:** Especially important when using the public/free Gemini API without billing enabled.

🌟 This conservative throttling ensures stable, reproducible evaluation results without interruptions.

----------------------------------------------------------------------
"""

import json
import time
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from config import GEMINI_API_KEY

# Import RAGAS components
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall, 
    answer_similarity
)
from ragas import evaluate
from ragas.llms.base import RunConfig
from utils.base_ragas_llm import GeminiLLMWrapper, GeminiClient
from utils.base_ragas_embeddings import HuggingfaceEmbeddings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_item(item: Dict[str, Any]) -> SingleTurnSample:
    """Convert a JSON item to a RAGAS SingleTurnSample."""
    query = item.get("query", item.get("question", ""))

    contexts = item.get("context", [])
    if not isinstance(contexts, list):
        contexts = [contexts]
    
    ground_truth = item.get("ground_truth", [])
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    
    return SingleTurnSample(
        user_input=query,
        retrieved_contexts=contexts,
        reference_contexts=ground_truth,
        response=item.get("answer", ""),
        reference=" ".join(ground_truth)
    )

def evaluate_single_metric(
    dataset: EvaluationDataset,
    metric,
    metric_name: str,
    llm_wrapper,
    embeddings,
    sample_idx: int
) -> Optional[float]:
    """
    Evaluate a single metric on a dataset.
    
    Args:
        dataset: The dataset to evaluate
        metric: The metric to evaluate
        metric_name: The name of the metric
        llm_wrapper: The LLM wrapper to use
        embeddings: The embeddings to use
        sample_idx: The index of the current sample
        
    Returns:
        The metric score or None if evaluation failed
    """
    try:
        print(f"💯 Evaluating {metric_name} for sample {sample_idx}...")
        result = evaluate(
            dataset,
            metrics=[metric],
            llm=llm_wrapper,
            embeddings=embeddings,
            raise_exceptions=False,
            show_progress=True
        )
        
        metric_value = result.to_pandas()[metric_name].iloc[0]
        print(f"✅ {metric_name}: {metric_value:.4f}")
        return metric_value
        
    except Exception as e:
        print(f"❌ Error evaluating {metric_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_rag_results(
    data_path: str = "results/qa_results.json",
    output_path: str = "results/ragas_scores.csv",
    gemini_api_key: str = "AIzaSyBJIvnOnSmp2z9XWZJOZHL8XZGust9x7Vk", 
    model_name: str = "gemini-2.0-flash",
    delay_seconds: int = 5
) -> pd.DataFrame:
    """
    Evaluate RAG results using RAGAS metrics.
    
    Args:
        data_path: Path to the JSON file with RAG results
        output_path: Path to save the evaluation results CSV
        gemini_api_key: Google AI API key
        model_name: Gemini model name
        delay_seconds: Delay between evaluations to avoid rate limits
    
    Returns:
        DataFrame with evaluation results
    """
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # If data is a single item, wrap it in a list
        data = [data]
        
    print(f"Found {len(data)} samples to evaluate")
    
    hf_embeddings = HuggingfaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    gemini_client = GeminiClient(
        api_key=gemini_api_key,
        model_name=model_name,
        temperature=0.3
    )
    llm_wrapper = GeminiLLMWrapper(gemini_client, retry_attempts=3, retry_delay=2)
    run_config = RunConfig()
    llm_wrapper.set_run_config(run_config)
    
    # Define metrics to evaluate
    metrics_to_evaluate = [
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
        ("context_precision", context_precision),
        ("context_recall", context_recall),
        ("answer_similarity", answer_similarity)
    ]
    
    all_results = []
    
    for idx, item in enumerate(data):
        print(f"\n🔍 Processing sample {idx+1}/{len(data)}: {item.get('query', '')[:60]}...")
        
        try:
            sample = process_item(item)
            sample_dataset = EvaluationDataset([sample])
            sample_results = {"query": item.get("query", "")}
            for metric_name, metric in metrics_to_evaluate:
                metric_value = evaluate_single_metric(
                    sample_dataset,
                    metric,
                    metric_name,
                    llm_wrapper,
                    hf_embeddings,
                    idx+1
                )
                sample_results[metric_name] = metric_value
        
                time.sleep(3) # Adding a short delay
        
            all_results.append(sample_results)
            
            if all_results:
                pd.DataFrame(all_results).to_csv(f"{output_path}.partial", index=False)
        
            if idx < len(data) - 1:
                print(f"⏱️ Waiting {delay_seconds} seconds before next sample...")
                time.sleep(delay_seconds)
                
        except Exception as e:
            print(f"❌ Error processing sample {idx+1}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("No results to save")
        return pd.DataFrame()
        
    final_df = pd.DataFrame(all_results)
    
    print("\n Average Scores:")
    for col in final_df.select_dtypes(include=['float64']).columns:
        values = final_df[col].dropna()
        if not values.empty:
            print(f"{col}: {values.mean():.4f}")
        else:
            print(f"{col}: No valid results")
    
    # Save to CSV file
    final_df.to_csv(output_path, index=False)
    print(f"\n📄 Saved full evaluation to {output_path}")
    
    return final_df

def main():
    
    try:
        evaluate_rag_results(
            gemini_api_key=GEMINI_API_KEY,
            delay_seconds=10 
        )
    except Exception as e:
        print(f"❌ Critical error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()