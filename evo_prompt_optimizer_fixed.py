"""
Evolutionary Hybrid Prompt Optimization Framework
Combines Genetic Algorithms (DEAP) with LLM-guided operators for prompt engineering.

Key Methods Inspired:
- EvoPrompt: LLM as evolutionary operators (mutation/crossover)
- CAPO: Racing strategy for efficient fitness evaluation
- GAAPO: Multi-objective optimization with NSGA-II

Author: AI-assisted generation for research/development
"""

import random
import json
import time
import copy
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# Third-party imports
from deap import base, creator, tools, algorithms
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PromptIndividual:
    """Container for an individual prompt in the population"""
    text: str
    fitness_scores: Tuple[float, float] = field(default=(0.0, 0.0))  # (accuracy, -avg_tokens)
    evaluations: int = 0


class EvoPromptOptimizer:
    """
    Evolutionary optimizer for LLM prompts using hybrid GA+LLM approach.

    Multi-objective optimization (NSGA-II):
    - Maximize accuracy on validation set
    - Minimize token cost (encoded as negative for maximization)

    Efficiency via racing:
    - Initial evaluation on subset (first N examples)
    - Discard poor performers early
    - Full evaluation only for survivors
    """

    def __init__(
        self,
        api_key: str,
        task_description: str,
        validation_set: List[Dict[str, str]],
        api_base: str = "https://api.x.ai/v1",  # xAI/Grok endpoint
        model: str = "grok-beta",
        population_size: int = 30,
        n_generations: int = 15,
        racing_subset_size: int = 10,
        racing_survival_rate: float = 0.5,
        mutation_rate: float = 0.5,
        crossover_rate: float = 0.5,
        seed_prompts: Optional[List[str]] = None,
        max_retries: int = 3,
        request_delay: float = 0.5  # Delay between API calls to avoid rate limits
    ):
        """
        Initialize the evolutionary prompt optimizer.

        Args:
            api_key: API key for LLM provider
            task_description: Description of the task (e.g., "math reasoning")
            validation_set: List of dicts with 'input' and 'output' keys
            api_base: API endpoint URL (default: xAI)
            model: Model name to use
            population_size: Number of prompts in population
            n_generations: Number of evolutionary generations
            racing_subset_size: Number of examples for initial racing evaluation
            racing_survival_rate: Fraction of population to keep after racing
            mutation_rate: Probability of mutation operator
            crossover_rate: Probability of crossover operator
            seed_prompts: Optional initial prompts (else auto-generated)
            max_retries: Max retries for API calls
            request_delay: Delay between API requests in seconds
        """
        self.api_key = api_key
        self.task_description = task_description
        self.validation_set = validation_set
        self.api_base = api_base
        self.model = model
        self.population_size = population_size
        self.n_generations = n_generations
        self.racing_subset_size = min(racing_subset_size, len(validation_set))
        self.racing_survival_rate = racing_survival_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed_prompts = seed_prompts or self._default_seed_prompts()
        self.max_retries = max_retries
        self.request_delay = request_delay

        # Initialize OpenAI client with xAI endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )

        # Statistics tracking
        self.total_api_calls = 0
        self.total_tokens_used = 0
        self.generation_stats = []

        # Setup DEAP framework for multi-objective optimization
        self._setup_deap()

    def _default_seed_prompts(self) -> List[str]:
        """Generate default seed prompts if none provided"""
        return [
            "Solve this problem step by step.",
            "Let's think through this carefully.",
            "Break down the problem into smaller steps and solve.",
            "Analyze the question and provide a clear solution.",
            "Think step-by-step to find the answer.",
            "Carefully reason through this problem.",
            "First, understand what's being asked, then solve systematically.",
            "Use logical reasoning to arrive at the correct answer.",
        ]

    def _setup_deap(self):
        """
        Configure DEAP framework for NSGA-II multi-objective optimization.

        Objectives:
        1. Maximize accuracy (weight=1.0)
        2. Maximize negative token cost (weight=1.0) - i.e., minimize cost
        """
        # Create fitness class (multi-objective maximization)
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))

        # Create individual class
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        # Initialize toolbox
        self.toolbox = base.Toolbox()

        # Register individual generator (wraps prompt text in list for DEAP compatibility)
        self.toolbox.register("individual", self._create_individual)

        # Register population generator
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register genetic operators (delegated to LLM)
        self.toolbox.register("mate", self._llm_crossover)
        self.toolbox.register("mutate", self._llm_mutate)

        # Register evaluation function with racing
        self.toolbox.register("evaluate", self._evaluate_with_racing)

        # Register NSGA-II selection
        self.toolbox.register("select", tools.selNSGA2)

        # Register clone operator (required for evolution loop)
        self.toolbox.register("clone", copy.deepcopy)

    def _create_individual(self) -> creator.Individual:
        """Create a DEAP individual containing a prompt"""
        if self.seed_prompts:
            prompt = random.choice(self.seed_prompts)
        else:
            # Generate prompt via LLM if no seeds left
            prompt = self._generate_initial_prompt()

        # DEAP expects list-like individuals
        individual = creator.Individual([prompt])
        return individual

    def _generate_initial_prompt(self) -> str:
        """Generate a new prompt using LLM for population diversity"""
        system_prompt = (
            f"You are an expert prompt engineer. Generate a high-quality instruction prompt "
            f"for the task: {self.task_description}. The prompt should guide an LLM to solve "
            f"problems accurately and efficiently. Output ONLY the prompt text, no explanations."
        )

        try:
            response = self._call_llm(
                system_prompt=system_prompt,
                user_prompt="Generate a new optimized prompt:",
                max_tokens=200
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"Failed to generate initial prompt via LLM: {e}. Using default.")
            return "Solve this problem carefully and show your work."

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Call LLM with retry logic and rate limiting.

        Args:
            system_prompt: System message for LLM
            user_prompt: User message/query
            max_tokens: Max tokens in response
            temperature: Sampling temperature

        Returns:
            LLM response text
        """
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                time.sleep(self.request_delay)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Track usage
                self.total_api_calls += 1
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens_used += response.usage.total_tokens

                return response.choices[0].message.content

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def _llm_mutate(self, individual: creator.Individual) -> Tuple[creator.Individual]:
        """
        LLM-guided mutation: Use LLM to refine/improve the prompt.

        Inspired by EvoPrompt mutation operator.

        Args:
            individual: DEAP individual (contains prompt in [0])

        Returns:
            Tuple containing mutated individual (DEAP convention)
        """
        original_prompt = individual[0]

        system_prompt = (
            f"You are an expert prompt engineer. Your task is to improve the following prompt "
            f"for {self.task_description}. Make it more effective while keeping it concise. "
            f"Output ONLY the improved prompt, no explanations or preamble."
        )

        user_prompt = f"Original prompt:\n{original_prompt}\n\nImprove this prompt:"

        try:
            # Get LLM-suggested mutation
            improved_prompt = self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=300,
                temperature=0.8  # Higher temp for diversity
            )

            # Self-reflection: Have LLM critique its own mutation
            if random.random() < 0.3:  # 30% chance of self-reflection
                improved_prompt = self._self_reflect(improved_prompt, original_prompt)

            individual[0] = improved_prompt.strip()

        except Exception as e:
            logger.warning(f"Mutation failed: {e}. Keeping original prompt.")
            # On failure, apply simple text mutation as fallback
            individual[0] = self._simple_text_mutation(original_prompt)

        return (individual,)

    def _llm_crossover(
        self,
        ind1: creator.Individual,
        ind2: creator.Individual
    ) -> Tuple[creator.Individual, creator.Individual]:
        """
        LLM-guided crossover: Use LLM to combine two prompts.

        Inspired by EvoPrompt crossover operator.

        Args:
            ind1, ind2: Parent individuals

        Returns:
            Tuple of two offspring individuals (DEAP convention)
        """
        prompt1 = ind1[0]
        prompt2 = ind2[0]

        system_prompt = (
            f"You are an expert prompt engineer. Combine the following two prompts for "
            f"{self.task_description} into a single, superior prompt that captures the "
            f"strengths of both. Output ONLY the combined prompt, no explanations."
        )

        user_prompt = (
            f"Prompt 1:\n{prompt1}\n\n"
            f"Prompt 2:\n{prompt2}\n\n"
            f"Create a combined prompt:"
        )

        try:
            # Get LLM-suggested crossover
            combined_prompt = self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=400,
                temperature=0.7
            )

            # Self-reflection on crossover result
            if random.random() < 0.25:  # 25% chance
                combined_prompt = self._self_reflect(combined_prompt, f"{prompt1} AND {prompt2}")

            # Both offspring get the combined prompt (could also create variants)
            ind1[0] = combined_prompt.strip()
            ind2[0] = combined_prompt.strip()

        except Exception as e:
            logger.warning(f"Crossover failed: {e}. Keeping original prompts.")
            # On failure, no crossover occurs

        return (ind1, ind2)

    def _self_reflect(self, new_prompt: str, context: str) -> str:
        """
        Self-reflection: Have LLM critique and optionally refine its generated prompt.

        Inspired by GAAPO self-refinement.

        Args:
            new_prompt: The prompt to critique
            context: Original context (for reference)

        Returns:
            Refined prompt (or original if no improvement)
        """
        system_prompt = (
            f"You are a critical evaluator of prompts for {self.task_description}. "
            f"Analyze the given prompt and suggest improvements if needed. If the prompt "
            f"is already excellent, return it unchanged. Output ONLY the final prompt."
        )

        user_prompt = (
            f"Prompt to evaluate:\n{new_prompt}\n\n"
            f"Critique this prompt and refine if necessary:"
        )

        try:
            refined = self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=350,
                temperature=0.6
            )
            return refined.strip()
        except Exception as e:
            logger.warning(f"Self-reflection failed: {e}. Using unreflected prompt.")
            return new_prompt

    def _simple_text_mutation(self, prompt: str) -> str:
        """Fallback simple text mutation if LLM fails"""
        mutations = [
            lambda p: p + " Think carefully.",
            lambda p: p + " Show your reasoning.",
            lambda p: "Carefully " + p.lower(),
            lambda p: p.replace(".", ", then verify your answer."),
            lambda p: p + " Double-check your work."
        ]
        return random.choice(mutations)(prompt)

    def _evaluate_with_racing(self, individual: creator.Individual) -> Tuple[float, float]:
        """
        CAPO-inspired racing evaluation for efficiency.

        Process:
        1. Evaluate on racing_subset_size examples
        2. If performance is poor, return early (avoid full evaluation)
        3. Otherwise, evaluate on full validation set

        Args:
            individual: Individual to evaluate

        Returns:
            Tuple of (accuracy, -avg_token_cost) for multi-objective optimization
        """
        prompt = individual[0]

        # Phase 1: Racing evaluation on subset
        racing_subset = self.validation_set[:self.racing_subset_size]
        racing_correct, racing_tokens = self._evaluate_on_dataset(prompt, racing_subset)
        racing_accuracy = racing_correct / len(racing_subset) if racing_subset else 0.0

        # Early stopping threshold (configurable)
        # If accuracy is very low on racing subset, skip full evaluation
        if racing_accuracy < 0.3 and len(racing_subset) > 5:
            avg_tokens = racing_tokens / len(racing_subset) if racing_subset else 0
            logger.debug(f"Racing: Early stop. Accuracy={racing_accuracy:.2f}")
            return (racing_accuracy, -avg_tokens)

        # Phase 2: Full evaluation on entire validation set
        full_correct, full_tokens = self._evaluate_on_dataset(prompt, self.validation_set)
        full_accuracy = full_correct / len(self.validation_set) if self.validation_set else 0.0
        avg_tokens = full_tokens / len(self.validation_set) if self.validation_set else 0

        logger.debug(
            f"Full eval: Accuracy={full_accuracy:.3f}, "
            f"Avg tokens={avg_tokens:.1f}, Prompt='{prompt[:50]}...'"
        )

        # Return fitness tuple (accuracy, -cost)
        # Negative cost so maximization optimizes for both high accuracy and low cost
        return (full_accuracy, -avg_tokens)

    def _evaluate_on_dataset(
        self,
        prompt: str,
        dataset: List[Dict[str, str]]
    ) -> Tuple[int, int]:
        """
        Evaluate a prompt on a dataset.

        Args:
            prompt: Instruction prompt to prepend to each example
            dataset: List of examples with 'input' and 'output'

        Returns:
            Tuple of (num_correct, total_tokens)
        """
        correct = 0
        total_tokens = 0

        for example in dataset:
            input_text = example['input']
            expected_output = example['output'].strip().lower()

            # Construct full prompt
            full_prompt = f"{prompt}\n\nProblem: {input_text}\n\nSolution:"

            try:
                # Get LLM response
                response = self._call_llm(
                    system_prompt="You are a helpful assistant.",
                    user_prompt=full_prompt,
                    max_tokens=512,
                    temperature=0.0  # Deterministic for evaluation
                )

                # Extract answer (simple string matching; could use regex for math)
                # For math, typically look for final number
                predicted = self._extract_answer(response).strip().lower()

                # Check correctness
                if predicted == expected_output or expected_output in predicted:
                    correct += 1

                # Estimate tokens (rough approximation: 4 chars per token)
                total_tokens += len(response) // 4

            except Exception as e:
                logger.warning(f"Evaluation error on example: {e}")
                # Count as incorrect on error
                continue

        return correct, total_tokens

    def _extract_answer(self, response: str) -> str:
        """
        Extract final answer from LLM response.

        Simple heuristic: Look for last number or statement.
        For production, use regex or parse structured output.

        Args:
            response: LLM response text

        Returns:
            Extracted answer string
        """
        # Try to find answer after common markers
        markers = ["answer:", "answer is", "=", "final answer:"]
        for marker in markers:
            if marker in response.lower():
                parts = response.lower().split(marker)
                if len(parts) > 1:
                    # Return everything after marker, cleaned
                    return parts[-1].strip().split()[0] if parts[-1].strip() else response

        # Fallback: return last line or whole response
        lines = response.strip().split('\n')
        return lines[-1].strip() if lines else response

    def optimize(self) -> Tuple[str, Dict[str, Any]]:
        """
        Run evolutionary optimization to find best prompt.

        Returns:
            Tuple of (best_prompt, statistics_dict)
        """
        logger.info(f"Starting evolutionary optimization for {self.n_generations} generations...")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Task: {self.task_description}")
        logger.info(f"Validation set size: {len(self.validation_set)}")

        # Initialize population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        logger.info("Evaluating initial population...")
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Track hall of fame (best individuals across all generations)
        hof = tools.ParetoFront()
        hof.update(population)

        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # Track accuracy
        stats.register("avg", lambda x: sum(x) / len(x) if x else 0)
        stats.register("max", max)
        stats.register("min", min)

        # Evolution loop
        for gen in range(self.n_generations):
            logger.info(f"\n=== Generation {gen + 1}/{self.n_generations} ===")

            # Select next generation using NSGA-II (Pareto-based selection)
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate individuals with invalid fitness (modified by operators)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            logger.info(f"Evaluating {len(invalid_ind)} modified individuals...")

            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            population[:] = offspring

            # Update hall of fame
            hof.update(population)

            # Log generation statistics
            record = stats.compile(population)
            self.generation_stats.append(record)

            logger.info(f"Accuracy - Max: {record['max']:.3f}, Avg: {record['avg']:.3f}, Min: {record['min']:.3f}")
            logger.info(f"API calls so far: {self.total_api_calls}, Tokens: {self.total_tokens_used}")

            # Log best prompts from Pareto front
            if hof:
                best = max(hof, key=lambda ind: ind.fitness.values[0])
                logger.info(f"Current best accuracy: {best.fitness.values[0]:.3f}")
                logger.info(f"Best prompt preview: '{best[0][:100]}...'")

        # Final selection: Pick best prompt from Pareto front
        # Prioritize accuracy, but consider token cost
        if not hof:
            logger.warning("No valid individuals in Pareto front. Using last population.")
            hof = population

        # Select individual with highest accuracy from Pareto front
        best_individual = max(hof, key=lambda ind: ind.fitness.values[0])
        best_prompt = best_individual[0]
        best_fitness = best_individual.fitness.values

        # Compile statistics
        stats_dict = {
            "best_prompt": best_prompt,
            "best_accuracy": best_fitness[0],
            "best_avg_tokens": -best_fitness[1],  # Convert back to positive
            "total_generations": self.n_generations,
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
            "generation_stats": self.generation_stats,
            "pareto_front_size": len(hof)
        }

        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Best accuracy: {best_fitness[0]:.3f}")
        logger.info(f"Average tokens: {-best_fitness[1]:.1f}")
        logger.info(f"Total API calls: {self.total_api_calls}")
        logger.info(f"Total tokens used: {self.total_tokens_used}")
        logger.info(f"\nBest prompt:\n{best_prompt}")
        logger.info(f"{'='*60}\n")

        return best_prompt, stats_dict


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Sample validation set (GSM8K-style math problems)
    # In production, load from actual dataset
    SAMPLE_VALIDATION_SET = [
        {
            "input": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "output": "18"
        },
        {
            "input": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "output": "3"
        },
        {
            "input": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "output": "70000"
        },
        {
            "input": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
            "output": "540"
        },
        {
            "input": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
            "output": "20"
        }
    ]

    # Configuration
    # NOTE: Replace with your actual xAI API key
    API_KEY = "your-xai-api-key-here"  # Get from https://console.x.ai

    # Initialize optimizer
    optimizer = EvoPromptOptimizer(
        api_key=API_KEY,
        task_description="solving grade school math word problems step-by-step",
        validation_set=SAMPLE_VALIDATION_SET,
        api_base="https://api.x.ai/v1",  # xAI/Grok endpoint
        model="grok-beta",  # Use Grok model
        population_size=20,  # Smaller for demo; use 30-50 for production
        n_generations=10,  # Reduced for demo; use 15-20+ for production
        racing_subset_size=3,  # Use first 3 examples for racing
        racing_survival_rate=0.5,
        mutation_rate=0.5,
        crossover_rate=0.5,
        request_delay=0.5  # Adjust based on rate limits
    )

    # Run optimization
    try:
        best_prompt, stats = optimizer.optimize()

        # Output results
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"\nBest Prompt Found:\n{best_prompt}")
        print(f"\nAccuracy: {stats['best_accuracy']:.2%}")
        print(f"Avg Token Cost: {stats['best_avg_tokens']:.1f}")
        print(f"Total API Calls: {stats['total_api_calls']}")
        print(f"Total Tokens: {stats['total_tokens_used']:,}")

        # Optionally save results to JSON
        output_file = "evo_prompt_results.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to {output_file}")

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise
