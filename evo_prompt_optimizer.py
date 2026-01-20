#!/usr/bin/env python3
"""
Evolutionary Hybrid Framework for LLM Prompt Optimization

This framework implements a state-of-the-art evolutionary algorithm for optimizing
LLM prompts, combining techniques from:
- EvoPrompt: LLM-based genetic operators (mutation/crossover)
- CAPO: Racing-based efficiency (partial evaluation to discard poor candidates)
- GAAPO: Multi-objective optimization with NSGA-II and self-reflection

Author: AI-Generated Evolutionary Framework
Dependencies: pip install deap openai
"""

import random
import json
import time
from typing import List, Dict, Tuple, Optional
from deap import base, creator, tools, algorithms
import openai


class EvoPromptOptimizer:
    """
    Evolutionary Prompt Optimizer using Genetic Algorithms (DEAP) and LLM-based operators.

    This framework implements hybrid evolutionary algorithms combining:
    - EvoPrompt: Using LLMs as evolutionary operators (mutation/crossover)
    - CAPO: Racing-based efficiency (partial eval to discard poor candidates early)
    - GAAPO: Multi-objective optimization with NSGA-II for Pareto selection

    The goal is to evolve prompts that maximize task accuracy while minimizing token cost.
    """

    def __init__(
        self,
        task_desc: str,
        val_set: List[Dict],
        api_key: str,
        api_base: str = "https://api.x.ai/v1",
        model: str = "grok-beta",
        population_size: int = 30,
        num_generations: int = 15,
        racing_threshold: int = 10,
        racing_keep_ratio: float = 0.5,
        mutation_prob: float = 0.5,
        crossover_prob: float = 0.5,
        seed_prompts: Optional[List[str]] = None
    ):
        """
        Initialize the evolutionary prompt optimizer.

        Args:
            task_desc: Description of the task (e.g., "math reasoning")
            val_set: Validation set with 'input' and 'output' keys
            api_key: API key for LLM
            api_base: API base URL (default: xAI/Grok)
            model: Model name (default: "grok-beta")
            population_size: Size of population (20-50 recommended)
            num_generations: Number of evolution generations (10-20 recommended)
            racing_threshold: Number of examples for initial racing eval
            racing_keep_ratio: Ratio of candidates to keep after racing (0.5 = 50%)
            mutation_prob: Probability of mutation operator
            crossover_prob: Probability of crossover operator
            seed_prompts: Optional list of seed prompts to start with
        """
        self.task_desc = task_desc
        self.val_set = val_set
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.population_size = population_size
        self.num_generations = num_generations
        self.racing_threshold = min(racing_threshold, len(val_set))
        self.racing_keep_ratio = racing_keep_ratio
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

        # Configure OpenAI client for xAI/Grok or other OpenAI-compatible providers
        openai.api_key = api_key
        openai.api_base = api_base

        # Initialize seed prompts (manual + LLM-generated)
        self.seed_prompts = seed_prompts or self._generate_seed_prompts()

        # DEAP setup for multi-objective optimization (NSGA-II)
        self._setup_deap()

        # Tracking for logging and cost analysis
        self.generation_stats = []
        self.total_api_calls = 0
        self.total_tokens = 0

    def _setup_deap(self):
        """
        Setup DEAP framework for multi-objective evolutionary algorithm.
        Using NSGA-II (Non-dominated Sorting Genetic Algorithm II) for Pareto-based selection.

        Multi-objective fitness:
        - Objective 1: Maximize accuracy (weight = 1.0)
        - Objective 2: Minimize average token cost (weight = -1.0)
        """
        # Create fitness class (maximize accuracy, minimize token cost)
        # weights: (1.0, -1.0) means maximize first objective, minimize second
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Register individual as a prompt string wrapped in list
        # (DEAP requires list-like individuals for genetic operators)
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register evolutionary operators
        self.toolbox.register("evaluate", self._evaluate_prompt)
        self.toolbox.register("mate", self._crossover_llm)
        self.toolbox.register("mutate", self._mutate_llm)
        self.toolbox.register("select", tools.selNSGA2)  # NSGA-II for multi-objective

    def _create_individual(self) -> creator.Individual:
        """
        Create an individual (prompt) from seed prompts.

        Returns:
            DEAP Individual containing a prompt string
        """
        prompt = random.choice(self.seed_prompts)
        ind = creator.Individual([prompt])
        return ind

    def _generate_seed_prompts(self) -> List[str]:
        """
        Generate initial seed prompts: mix of manual and LLM-generated.

        This implements the initialization strategy from EvoPrompt and GAAPO,
        combining human-designed prompts (based on prompt engineering best practices)
        with LLM-generated variations for diversity.

        Returns:
            List of seed prompt strings
        """
        # Manual seed prompts (based on common prompt engineering patterns)
        # These represent diverse starting points for evolution
        manual_seeds = [
            "Think step-by-step to solve this problem.",
            "Let's solve this carefully and show all work.",
            "Break down the problem and solve it step by step.",
            "Analyze the problem and provide a detailed solution.",
            "Solve this problem systematically, showing each step.",
            "Let's approach this problem methodically.",
            "Carefully work through this problem step by step.",
            "Think through this problem logically and solve it.",
            "First understand the problem, then solve it step-by-step.",
            "Use clear reasoning to find the answer.",
        ]

        # Generate additional prompts via LLM for diversity
        # This creates varied starting points beyond human templates
        llm_seeds = []
        generation_prompts = [
            f"Generate an effective instruction prompt for {self.task_desc} tasks. Output only the prompt, no explanation.",
            f"Create a concise prompt that guides an AI to excel at {self.task_desc}. Output only the prompt.",
            f"Write a short instruction for solving {self.task_desc} problems accurately. Output only the prompt.",
            f"Design a prompt that ensures careful reasoning for {self.task_desc}. Output only the prompt.",
        ]

        for gen_prompt in generation_prompts:
            try:
                response = self._call_llm(gen_prompt, max_tokens=100, temperature=0.9)
                if response and len(response.strip()) > 5:
                    llm_seeds.append(response.strip())
            except Exception as e:
                print(f"Warning: Failed to generate LLM seed: {e}")

        # Combine manual and LLM seeds
        all_seeds = manual_seeds + llm_seeds
        print(f"Initialized {len(all_seeds)} seed prompts ({len(manual_seeds)} manual, {len(llm_seeds)} LLM-generated)")
        return all_seeds

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        retry_count: int = 3
    ) -> Optional[str]:
        """
        Call LLM API with retry logic and error handling.

        Handles common API issues:
        - Rate limits (with exponential backoff)
        - Transient errors (with retry)
        - Token tracking for cost monitoring

        Args:
            prompt: Input prompt for the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            retry_count: Number of retries on failure

        Returns:
            Generated text or None on failure
        """
        for attempt in range(retry_count):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=30
                )

                # Track usage for cost monitoring
                self.total_api_calls += 1
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens += response.usage.total_tokens

                return response.choices[0].message.content

            except openai.error.RateLimitError:
                # Exponential backoff for rate limits
                wait_time = (2 ** attempt) * 2
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)

            except openai.error.APIError as e:
                print(f"API error (attempt {attempt+1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2)

            except Exception as e:
                print(f"Unexpected error calling LLM: {e}")
                return None

        return None

    def _evaluate_prompt_on_example(self, prompt: str, example: Dict) -> bool:
        """
        Evaluate a prompt on a single example from the validation set.

        Args:
            prompt: The instruction prompt to test
            example: Dict with 'input' (problem) and 'output' (expected answer)

        Returns:
            True if the LLM's answer matches expected output, False otherwise
        """
        # Construct full prompt: instruction + task input
        full_prompt = f"{prompt}\n\nProblem: {example['input']}\n\nAnswer:"

        try:
            response = self._call_llm(full_prompt, max_tokens=256, temperature=0.0)
            if not response:
                return False

            # Extract answer (simple matching for demo; enhance for production)
            # For production: use more robust answer extraction (regex, parsing, etc.)
            response = response.strip().lower()
            expected = str(example['output']).strip().lower()

            # Check if expected answer appears in response
            # This handles cases where LLM outputs "The answer is 8" vs just "8"
            return expected in response

        except Exception as e:
            print(f"Error evaluating example: {e}")
            return False

    def _evaluate_prompt(self, individual: creator.Individual) -> Tuple[float, float]:
        """
        Evaluate fitness of a prompt using racing strategy (CAPO-inspired).

        Multi-objective fitness evaluation:
        1. Accuracy on validation set (maximize)
        2. Average token cost per example (minimize)

        Racing strategy for efficiency:
        - First, evaluate on small subset (racing_threshold examples)
        - Discard poor performers early (those below threshold)
        - Full evaluation only on promising candidates
        - This reduces API calls by ~50% without sacrificing quality (CAPO paper)

        Args:
            individual: DEAP individual containing prompt string

        Returns:
            Tuple of (accuracy, avg_token_cost)
        """
        prompt = individual[0]

        # === Racing Phase: Quick evaluation on subset ===
        # Evaluate on first N examples to identify poor candidates early
        racing_examples = self.val_set[:self.racing_threshold]
        racing_correct = sum(
            self._evaluate_prompt_on_example(prompt, ex)
            for ex in racing_examples
        )
        racing_accuracy = racing_correct / len(racing_examples)

        # If racing accuracy is too low, skip full eval (efficiency optimization)
        # This implements CAPO's racing mechanism: discard bottom performers early
        # Threshold of 0.3 means prompts getting <30% on racing set are rejected
        if racing_accuracy < 0.3:
            # Estimate token cost without full evaluation
            # Based on prompt length (rough approximation)
            estimated_tokens = len(prompt.split()) * 1.5
            return (racing_accuracy, estimated_tokens)

        # === Full Evaluation Phase ===
        # Candidate passed racing, now evaluate on full validation set
        correct = 0
        token_count = 0

        for example in self.val_set:
            is_correct = self._evaluate_prompt_on_example(prompt, example)
            if is_correct:
                correct += 1

            # Track token cost (approximate based on prompt + input length)
            # In production, use actual token counts from API response.usage
            token_count += len(prompt.split()) + len(example['input'].split())

        accuracy = correct / len(self.val_set)
        avg_tokens = token_count / len(self.val_set)

        return (accuracy, avg_tokens)

    def _mutate_llm(self, individual: creator.Individual) -> Tuple[creator.Individual]:
        """
        Mutation operator using LLM to refine/improve prompt.

        This implements LLM-based mutation from EvoPrompt, where the LLM
        acts as a creative operator to modify prompts for better performance.
        Unlike traditional GA mutation (random bit flips), this uses the LLM's
        understanding of language and prompting to make intelligent modifications.

        Args:
            individual: Individual to mutate

        Returns:
            Tuple containing mutated individual (DEAP convention)
        """
        original_prompt = individual[0]

        # Meta-prompt for mutation: ask LLM to improve the prompt
        # This leverages the LLM's knowledge of effective prompting strategies
        mutation_meta_prompt = f"""You are a prompt engineering expert. Your task is to improve the following instruction prompt for {self.task_desc} tasks.

Current prompt: "{original_prompt}"

Generate an improved version that will lead to better accuracy. Focus on clarity, specificity, and effective reasoning guidance. Output ONLY the improved prompt, no explanation.

Improved prompt:"""

        try:
            improved_prompt = self._call_llm(mutation_meta_prompt, max_tokens=150, temperature=0.8)

            if improved_prompt and len(improved_prompt.strip()) > 5:
                # Apply self-reflection: critique and refine the generated prompt
                # This implements GAAPO's self-reflection mechanism
                improved_prompt = self._self_reflect(improved_prompt)
                individual[0] = improved_prompt.strip()
            else:
                # Fallback: minor variation if LLM fails
                individual[0] = original_prompt + " Be precise and thorough."

        except Exception as e:
            print(f"Mutation error: {e}")
            # Keep original on error

        return (individual,)

    def _crossover_llm(
        self,
        ind1: creator.Individual,
        ind2: creator.Individual
    ) -> Tuple[creator.Individual, creator.Individual]:
        """
        Crossover operator using LLM to combine two prompts.

        This implements LLM-based crossover from EvoPrompt and GAAPO,
        where the LLM intelligently merges two parent prompts.
        Unlike traditional crossover (cutting/splicing strings), this uses
        semantic understanding to combine the best aspects of both parents.

        Args:
            ind1: First parent individual
            ind2: Second parent individual

        Returns:
            Tuple of two offspring individuals
        """
        prompt1 = ind1[0]
        prompt2 = ind2[0]

        # Meta-prompt for crossover: ask LLM to combine prompts
        # The LLM identifies complementary strengths and merges them
        crossover_meta_prompt = f"""You are a prompt engineering expert. Combine the best elements of these two instruction prompts for {self.task_desc} tasks:

Prompt 1: "{prompt1}"
Prompt 2: "{prompt2}"

Create a new prompt that merges their strengths while maintaining clarity and effectiveness. Output ONLY the combined prompt, no explanation.

Combined prompt:"""

        try:
            combined_prompt = self._call_llm(crossover_meta_prompt, max_tokens=150, temperature=0.7)

            if combined_prompt and len(combined_prompt.strip()) > 5:
                # Apply self-reflection to refine the combination
                combined_prompt = self._self_reflect(combined_prompt)

                # Create first offspring with the combined prompt
                ind1[0] = combined_prompt.strip()

                # Second offspring: create a variation for diversity
                # This maintains genetic diversity in the population
                variation_prompt = f"Rephrase this prompt slightly while keeping the same meaning: {combined_prompt}\n\nRephrased:"
                ind2_prompt = self._call_llm(variation_prompt, max_tokens=150, temperature=0.6)
                ind2[0] = ind2_prompt.strip() if ind2_prompt else combined_prompt.strip()

        except Exception as e:
            print(f"Crossover error: {e}")
            # Keep originals on error

        return ind1, ind2

    def _self_reflect(self, prompt: str) -> str:
        """
        Self-reflection: LLM critiques and optionally refines the prompt.

        This implements the self-reflection mechanism from GAAPO, where
        generated prompts are critiqued and refined iteratively.
        This acts as a quality control step to catch issues like:
        - Unclear instructions
        - Excessive verbosity
        - Missing key reasoning guidance

        Args:
            prompt: Prompt to reflect on

        Returns:
            Refined prompt (or original if already good)
        """
        reflection_meta_prompt = f"""Review this instruction prompt for {self.task_desc} tasks and improve it if needed:

Prompt: "{prompt}"

If the prompt is already excellent, output it as-is. If it has issues (unclear, too verbose, missing key guidance), output an improved version. Output ONLY the final prompt, no explanation.

Final prompt:"""

        try:
            refined = self._call_llm(reflection_meta_prompt, max_tokens=150, temperature=0.5)
            return refined.strip() if refined and len(refined.strip()) > 5 else prompt
        except Exception as e:
            print(f"Self-reflection error: {e}")
            return prompt

    def evolve(self) -> str:
        """
        Main evolution loop: Run evolutionary algorithm to optimize prompts.

        This implements the core EA loop with:
        - NSGA-II selection for multi-objective optimization (Pareto fronts)
        - LLM-based mutation and crossover operators
        - Racing for computational efficiency
        - Comprehensive logging for transparency and debugging

        Returns:
            Best prompt from final Pareto front (highest accuracy)
        """
        print(f"{'='*60}")
        print(f"Starting Evolutionary Prompt Optimization")
        print(f"{'='*60}")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Task: {self.task_desc}")
        print(f"Validation set: {len(self.val_set)} examples")
        print(f"Racing threshold: {self.racing_threshold} examples")
        print(f"{'='*60}\n")

        # === Initialize Population ===
        # Create initial population from seed prompts
        population = self.toolbox.population(n=self.population_size)

        # === Evaluate Initial Population ===
        print("Evaluating initial population...")
        fitnesses = map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Log initial generation stats
        self._log_generation(0, population)

        # === Evolution Loop ===
        for gen in range(1, self.num_generations + 1):
            print(f"\n{'='*60}")
            print(f"Generation {gen}/{self.num_generations}")
            print(f"{'='*60}")

            # Select next generation using NSGA-II (Pareto-based selection)
            # This maintains diversity by selecting across Pareto fronts
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # === Apply Crossover ===
            # Pair up individuals and apply crossover with probability
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    # Invalidate fitness since individuals changed
                    del child1.fitness.values
                    del child2.fitness.values

            # === Apply Mutation ===
            # Mutate individuals with probability
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    # Invalidate fitness since individual changed
                    del mutant.fitness.values

            # === Evaluate Offspring ===
            # Only evaluate individuals with invalid fitness (those that were modified)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            print(f"Evaluating {len(invalid_ind)} new individuals...")
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # === Replace Population ===
            # Generational replacement: offspring become new population
            population[:] = offspring

            # Log generation statistics
            self._log_generation(gen, population)

        # === Extract Best Prompt ===
        # Get best prompt from final Pareto front (highest accuracy)
        best_prompt = self._get_best_prompt(population)

        # === Print Final Results ===
        print(f"\n{'='*60}")
        print("Evolution Complete!")
        print(f"{'='*60}")
        print(f"Total API calls: {self.total_api_calls}")
        print(f"Total tokens: {self.total_tokens}")
        print(f"\nBest Prompt (highest accuracy from Pareto front):")
        print(f"  \"{best_prompt['prompt']}\"")
        print(f"\nPerformance:")
        print(f"  Accuracy: {best_prompt['accuracy']:.2%}")
        print(f"  Avg tokens: {best_prompt['avg_tokens']:.1f}")
        print(f"{'='*60}\n")

        return best_prompt['prompt']

    def _log_generation(self, gen: int, population: List):
        """
        Log statistics for current generation.

        Tracks key metrics for monitoring evolution progress:
        - Max/mean accuracy (are we improving?)
        - Min/mean tokens (are we optimizing efficiency?)

        Args:
            gen: Generation number
            population: Current population
        """
        fits = [ind.fitness.values for ind in population]
        accuracies = [f[0] for f in fits]
        tokens = [f[1] for f in fits]

        stats = {
            'generation': gen,
            'max_accuracy': max(accuracies),
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'min_tokens': min(tokens),
            'mean_tokens': sum(tokens) / len(tokens),
        }

        self.generation_stats.append(stats)

        print(f"Statistics:")
        print(f"  Max accuracy: {stats['max_accuracy']:.2%}")
        print(f"  Mean accuracy: {stats['mean_accuracy']:.2%}")
        print(f"  Min tokens: {stats['min_tokens']:.1f}")
        print(f"  Mean tokens: {stats['mean_tokens']:.1f}")

    def _get_best_prompt(self, population: List) -> Dict:
        """
        Get best prompt from Pareto front (highest accuracy).

        In multi-objective optimization, there's no single "best" solution,
        but a Pareto front of non-dominated solutions (trade-offs).
        We select the prompt with highest accuracy from this front.

        Args:
            population: Final population

        Returns:
            Dict with best prompt and its fitness metrics
        """
        # Get Pareto front (non-dominated individuals)
        # These are solutions where no other solution is better in all objectives
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # Among Pareto front, select highest accuracy
        # (User can modify this to prefer different trade-offs)
        best_ind = max(pareto_front, key=lambda ind: ind.fitness.values[0])

        return {
            'prompt': best_ind[0],
            'accuracy': best_ind.fitness.values[0],
            'avg_tokens': best_ind.fitness.values[1],
        }


# ============================================================================
# Sample Usage with Dummy Math Reasoning Examples
# ============================================================================

if __name__ == "__main__":
    # Sample validation set: Math reasoning problems (GSM8K-style)
    # In production, use actual GSM8K dataset or your task-specific validation set
    sample_val_set = [
        {
            'input': 'Janet has 5 apples. She buys 3 more apples. How many apples does she have now?',
            'output': '8'
        },
        {
            'input': 'A book costs $12. If you buy 3 books, how much do you pay in total?',
            'output': '36'
        },
        {
            'input': 'Tom has 20 candies. He gives 7 to his friend. How many candies does Tom have left?',
            'output': '13'
        },
        {
            'input': 'A train travels 60 miles per hour. How far does it travel in 3 hours?',
            'output': '180'
        },
        {
            'input': 'Sarah has $50. She spends $18 on lunch and $12 on a book. How much money does she have left?',
            'output': '20'
        },
    ]

    # ========================================================================
    # Configuration
    # ========================================================================
    # NOTE: Replace with your actual API key for xAI/Grok or other provider
    # For xAI/Grok: Get API key from https://console.x.ai/
    # For OpenAI: Use api_base="https://api.openai.com/v1", model="gpt-4"
    # For other providers: Adjust api_base and model accordingly

    API_KEY = "your-api-key-here"  # REPLACE WITH ACTUAL KEY
    API_BASE = "https://api.x.ai/v1"  # xAI/Grok endpoint
    MODEL = "grok-beta"  # or "grok-2-latest", "gpt-4", etc.

    # For testing without actual API calls, you can:
    # 1. Mock the _call_llm method
    # 2. Use a local model via API-compatible server (e.g., vLLM, Ollama)
    # 3. Use a smaller validation set and fewer generations

    # ========================================================================
    # Initialize Optimizer
    # ========================================================================
    optimizer = EvoPromptOptimizer(
        task_desc="math reasoning",
        val_set=sample_val_set,
        api_key=API_KEY,
        api_base=API_BASE,
        model=MODEL,
        population_size=20,      # Use 30-50 for production
        num_generations=10,      # Use 15-20 for production
        racing_threshold=3,      # Use 10-20 for larger val_sets
        racing_keep_ratio=0.5,
        mutation_prob=0.5,
        crossover_prob=0.5,
    )

    # ========================================================================
    # Run Evolution
    # ========================================================================
    print("\nStarting evolutionary optimization...")
    print("Note: This will make API calls to the configured LLM provider.\n")

    best_prompt = optimizer.evolve()

    # ========================================================================
    # Save Results
    # ========================================================================
    results = {
        'best_prompt': best_prompt,
        'generation_stats': optimizer.generation_stats,
        'total_api_calls': optimizer.total_api_calls,
        'total_tokens': optimizer.total_tokens,
        'config': {
            'task': optimizer.task_desc,
            'population_size': optimizer.population_size,
            'num_generations': optimizer.num_generations,
            'model': optimizer.model,
        }
    }

    output_file = 'evo_prompt_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nYou can now use the best prompt in your applications!")
