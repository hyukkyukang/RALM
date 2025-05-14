import logging
import statistics
from typing import Any, Dict, List

import hkkang_utils.misc as misc_utils
import hkkang_utils.time as time_utils
import hydra
import torch
from omegaconf import DictConfig

from src.retrieval import SentenceTransformerRetriever
from src.tokenization import TOKENIZER_REGISTRY

logger = logging.getLogger("AnalyzeSearchSpeed")

# Changing the precision of the matmul operation to high causes error when torch compile the flex attention
torch._dynamo.config.cache_size_limit = 1000
torch.set_float32_matmul_precision("high")


def run_benchmark(
    searcher: SentenceTransformerRetriever,
    queries: List[str],
    k: int = 10,
    return_as_text: bool = False,
    num_runs: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Run benchmark tests on different search methods.

    Args:
        searcher: The searcher instance to use
        queries: List of queries to search for
        k: Number of results to return per query
        return_as_text: Whether to return results as text
        num_runs: Number of times to run each test for averaging

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    # Define the methods to test
    methods = {
        "single": lambda: [searcher.search(q, k, return_as_text) for q in queries],
        "batch": lambda: searcher.search_batch(queries, k, return_as_text),
    }
    warmed_up = False
    # Run benchmarks for each method
    for method_name, method_func in methods.items():
        logger.info(f"Benchmarking {method_name} search method...")

        # Warm-up run
        if not warmed_up:
            logger.info("Performing warm-up run...")
            method_func()
            warmed_up = True

        # Actual timed runs
        times = []
        for i in range(num_runs):
            logger.info(f"Run {i+1}/{num_runs}")
            with time_utils.Timer(f"{method_name}_search").measure() as timer:
                method_func()
            times.append(timer.elapsed_time)

        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0

        results[method_name] = {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_dev": std_dev,
            "times": times,
            "queries_per_second": len(queries) / avg_time,
        }

        logger.info(
            f"{method_name.capitalize()} search: {avg_time:.4f}s avg, {min_time:.4f}s min, {max_time:.4f}s max"
        )
        logger.info(f"Queries per second: {len(queries) / avg_time:.2f}")

    return results


def print_comparison_table(results: Dict[str, Dict[str, Any]], queries: List[str]):
    """Print a formatted comparison table of the benchmark results."""
    print("\n" + "=" * 80)
    print(f"SEARCH SPEED COMPARISON (for {len(queries)} queries)")
    print("=" * 80)
    print(
        f"{'Method':<10} | {'Avg Time (s)':<12} | {'Min Time (s)':<12} | {'Max Time (s)':<12} | {'Queries/s':<10} | {'Std Dev':<8}"
    )
    print("-" * 80)

    for method, stats in results.items():
        print(
            f"{method:<10} | "
            f"{stats['avg_time']:<12.4f} | "
            f"{stats['min_time']:<12.4f} | "
            f"{stats['max_time']:<12.4f} | "
            f"{stats['queries_per_second']:<10.2f} | "
            f"{stats['std_dev']:<8.4f}"
        )

    print("=" * 80)

    # Find the fastest method
    fastest_method = min(results.items(), key=lambda x: x[1]["avg_time"])
    print(f"Fastest method: {fastest_method[0]} ({fastest_method[1]['avg_time']:.4f}s)")

    # Calculate speedups
    baseline = results["single"]["avg_time"]
    for method, stats in results.items():
        if method != "single":
            speedup = baseline / stats["avg_time"]
            print(f"Speedup of {method} vs single: {speedup:.2f}x")

    print("=" * 80)


@hydra.main(version_base=None, config_path="/home/user/RALM/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Configure logging
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()

    # Load tokenizer
    tokenizer = TOKENIZER_REGISTRY[cfg.model.name].from_pretrained(cfg.model.base_name)

    # Define test queries - longer queries around 64 tokens when tokenized with Llama tokenizers
    queries = [
        "The Silk Road facilitated cultural exchange and trade across Eurasia for nearly two millennia. Please analyze the historical development of these trade networks, the major goods exchanged and their economic significance, how religious ideas and cultural practices spread along these routes, the political entities that controlled different segments of the Silk Road throughout history, and how this ancient network compares to modern global trade systems and the contemporary Belt and Road Initiative.",
        "I'm researching the historical significance of Paris as the capital of France. Can you provide detailed information about its establishment as the capital, major historical events that took place there, and how it has evolved politically and culturally over the centuries?",
        "George Orwell's novel '1984' is considered one of the most influential dystopian works of the 20th century. Could you analyze the major themes of totalitarianism, surveillance, and thought control in the book, and explain how these concepts relate to modern society and contemporary political developments?",
        "Einstein's theory of relativity revolutionized our understanding of space, time, and gravity. I'd like to understand both the special and general theories, their mathematical foundations, experimental validations throughout history, and how they've influenced modern physics and cosmology up to the present day.",
        "World War II was a global conflict with far-reaching consequences. Can you provide a comprehensive analysis of the major causes leading to the war, key turning points in different theaters of operation, the political alliances that shifted throughout the conflict, and the long-term geopolitical impact after its conclusion?",
        "Photosynthesis is a fundamental biological process that sustains life on Earth. Could you explain in detail the light-dependent and light-independent reactions, the role of chlorophyll and other pigments, how environmental factors affect photosynthetic efficiency, and its evolutionary development across different plant species?",
        "Climate change represents one of the most significant challenges facing humanity in the 21st century. Please provide an in-depth analysis of the primary anthropogenic and natural factors contributing to global warming, the observed and projected impacts on ecosystems worldwide, and the effectiveness of various mitigation strategies being implemented.",
        "Neil Armstrong's moon landing in 1969 marked a pivotal moment in human exploration. I'm interested in learning about the entire Apollo program that led to this achievement, including the technological innovations, political context during the Space Race, the specific challenges of the Apollo 11 mission, and how this accomplishment influenced subsequent space exploration.",
        "The theory of evolution through natural selection fundamentally changed our understanding of life on Earth. Could you explain Darwin's original concepts, how modern genetics has enhanced evolutionary theory, the mechanisms of speciation and adaptation, and how evolutionary biology continues to develop with new discoveries in genomics and paleontology?",
        "The human immune system is an incredibly complex network of cells, tissues, and organs that protects us from disease. Please provide a detailed explanation of both innate and adaptive immunity, the role of different immune cells like T-cells and B-cells, how vaccines leverage immune memory, and how autoimmune disorders and immunodeficiencies develop.",
        "Quantum mechanics represents a fundamental shift from classical physics in describing the behavior of matter and energy at the atomic and subatomic scales. Could you elaborate on wave-particle duality, Heisenberg's uncertainty principle, quantum entanglement, superposition, the various interpretations of quantum theory, and how these principles are applied in modern technologies?",
        "Artificial intelligence has rapidly evolved from theoretical concepts to practical applications that impact nearly every aspect of modern society. I'm interested in understanding the historical development of AI research, the fundamental algorithms and computational approaches that enable machine learning, current applications across different industries, ethical considerations regarding AI deployment, and how experts predict AI technologies will continue to evolve in the coming decades.",
        "The human brain is often described as the most complex structure in the known universe. Could you explain in detail the neuroanatomy of the brain, how different regions coordinate to process information, the current scientific understanding of consciousness and cognition, major neurotransmitter systems and their functions, and how modern neuroscience techniques have advanced our knowledge of brain function and neurological disorders?",
        "Biodiversity loss represents one of the most pressing environmental challenges of our time. Please provide an in-depth analysis of the primary causes of species extinction, the ecological consequences of reduced biodiversity in different ecosystems, how biodiversity is measured and monitored by scientists, international conservation efforts and their effectiveness, and the potential economic and social impacts of continued biodiversity decline.",
        "The ancient civilizations of Mesopotamia are credited with numerous innovations that formed the foundation of human civilization. I'd like to understand the development of early Mesopotamian city-states, their agricultural and irrigation systems, the invention of writing and its impact on record-keeping and knowledge transmission, their religious beliefs and practices, and how their political and legal structures influenced subsequent civilizations throughout the Mediterranean and Middle East.",
        "Cryptocurrency and blockchain technology have emerged as potentially transformative innovations in finance and digital record-keeping. Could you explain the fundamental cryptographic principles that enable blockchain systems, the differences between major cryptocurrencies and their underlying technologies, the environmental concerns associated with proof-of-work systems, regulatory challenges facing digital currencies, and potential applications of blockchain beyond financial transactions?",
        "The Great Barrier Reef is the world's largest coral reef system and a vital marine ecosystem currently facing numerous threats. Please provide a detailed explanation of the reef's formation and ecological structure, the diverse species that comprise its ecosystem, the specific impacts of climate change including coral bleaching events, other anthropogenic pressures affecting reef health, and conservation strategies being implemented to preserve this natural wonder for future generations.",
        "The Industrial Revolution fundamentally transformed economic systems, social structures, and daily life across Europe and North America. I'm interested in understanding the technological innovations that drove industrialization, how manufacturing processes evolved, the changing relationship between workers and employers, the environmental consequences of rapid industrialization, and how these historical developments continue to influence contemporary economic systems and labor practices.",
        "Nuclear fusion has long been pursued as a potentially limitless and clean energy source. Could you provide a comprehensive explanation of the physics principles behind nuclear fusion, the technical challenges preventing commercial fusion power, the different approaches being researched including tokamak and inertial confinement designs, recent breakthroughs in fusion research, and realistic timelines for when fusion might contribute significantly to global energy production?",
        "The Silk Road facilitated cultural exchange and trade across Eurasia for nearly two millennia. Please analyze the historical development of these trade networks, the major goods exchanged and their economic significance, how religious ideas and cultural practices spread along these routes, the political entities that controlled different segments of the Silk Road throughout history, and how this ancient network compares to modern global trade systems and the contemporary Belt and Road Initiative.",
    ]

    # Process queries to ensure they're exactly 64 tokens
    logger.info("Processing queries to ensure they're exactly 64 tokens...")
    modified_queries = []
    for query in queries:
        # Tokenize the query
        token_ids = tokenizer.encode(query, add_special_tokens=False)

        if len(token_ids) < 64:
            # If less than 64 tokens, repeat the first tokens to pad up to 64
            tokens_needed = 64 - len(token_ids)
            padding_tokens = token_ids[:tokens_needed]
            # Ensure we don't exceed 64 tokens if the padding is too long
            while len(token_ids) + len(padding_tokens) > 64:
                padding_tokens = padding_tokens[:-1]
            token_ids = token_ids + padding_tokens
        elif len(token_ids) > 64:
            # If more than 64 tokens, truncate
            token_ids = token_ids[:64]

        # Decode back to text
        modified_query = tokenizer.decode(token_ids)
        modified_queries.append(modified_query)

        # Log token count for verification
        logger.info(f"Query length: {len(token_ids)} tokens")

    queries = modified_queries
    # Repeat queries 50 times to test bigger batch search (of total size 1000)
    queries = queries * 50

    # Test parameters
    k = 5
    num_runs = 10

    # Test with different return_as_text settings
    all_results = {
        True: {},
        False: {},
    }  # Initialize the results dictionary outside the loop

    local_searcher = SentenceTransformerRetriever(cfg=cfg.retrieval, global_cfg=cfg)
    for return_as_text in [True, False]:
        print("\n" + "=" * 80)
        print(f"BENCHMARKING WITH return_as_text={return_as_text}")
        print("=" * 80)

        # Test with local searcher
        logger.info(
            f"Testing with local CustomSearcher (return_as_text={return_as_text})..."
        )
        local_results = run_benchmark(
            local_searcher,
            queries,
            k=k,
            return_as_text=return_as_text,
            num_runs=num_runs,
        )
        print_comparison_table(local_results, queries)

        # Store a copy of the results
        all_results[return_as_text]["local"] = local_results.copy()

    # Compare return_as_text=True vs return_as_text=False
    print("\n" + "=" * 80)
    print("RETURN AS TEXT TRUE VS FALSE COMPARISON")
    print("=" * 80)

    # Print comparison table for local searcher
    print("LOCAL SEARCHER - TRUE vs FALSE")
    print("-" * 80)
    print(
        f"{'Method':<10} | {'True (s)':<10} | {'False (s)':<10} | {'Difference':<10} | {'False/True':<12}"
    )
    print("-" * 80)

    for method in all_results[True]["local"].keys():
        true_time = all_results[True]["local"][method]["avg_time"]
        false_time = all_results[False]["local"][method]["avg_time"]
        diff = false_time - true_time
        ratio = false_time / true_time if true_time > 0 else float("inf")

        print(
            f"{method:<10} | "
            f"{true_time:<10.4f} | "
            f"{false_time:<10.4f} | "
            f"{diff:<10.4f} | "
            f"{ratio:<12.2f}x"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
