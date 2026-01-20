#!/usr/bin/env python3
"""
Example usage of EvoPromptOptimizer for different tasks.

This demonstrates how to use the evolutionary prompt optimizer
for various tasks and configurations.
"""

from evo_prompt_optimizer import EvoPromptOptimizer

# ============================================================================
# Example 1: Math Reasoning (GSM8K-style)
# ============================================================================

def example_math_reasoning():
    """Example: Optimizing prompts for math word problems."""

    # Larger validation set for better evaluation
    math_val_set = [
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
        {
            'input': 'A baker makes 24 cookies. He sells them in boxes of 6. How many boxes can he make?',
            'output': '4'
        },
        {
            'input': 'If a car uses 8 gallons of gas to travel 200 miles, how many gallons does it use per mile?',
            'output': '0.04'
        },
        {
            'input': 'A school has 450 students. If 30% are in the science club, how many students are in the science club?',
            'output': '135'
        },
        {
            'input': 'John runs 5 miles every day. How many miles does he run in 2 weeks?',
            'output': '70'
        },
        {
            'input': 'A pizza is cut into 8 slices. If 3 people share it equally, how many slices does each person get?',
            'output': '2.67'
        },
    ]

    optimizer = EvoPromptOptimizer(
        task_desc="math reasoning",
        val_set=math_val_set,
        api_key="your-api-key-here",  # REPLACE
        api_base="https://api.x.ai/v1",
        model="grok-beta",
        population_size=25,
        num_generations=12,
        racing_threshold=5,
    )

    best_prompt = optimizer.evolve()
    return best_prompt


# ============================================================================
# Example 2: Code Generation
# ============================================================================

def example_code_generation():
    """Example: Optimizing prompts for Python code generation."""

    code_val_set = [
        {
            'input': 'Write a function to check if a number is prime.',
            'output': 'def is_prime'
        },
        {
            'input': 'Write a function to reverse a string.',
            'output': 'def reverse_string'
        },
        {
            'input': 'Write a function to find the factorial of a number.',
            'output': 'def factorial'
        },
        {
            'input': 'Write a function to check if a string is a palindrome.',
            'output': 'def is_palindrome'
        },
        {
            'input': 'Write a function to find the maximum element in a list.',
            'output': 'def find_max'
        },
    ]

    # Custom seed prompts for code generation
    code_seed_prompts = [
        "Write clean, well-documented Python code.",
        "Implement the function with proper error handling.",
        "Create an efficient Python function with docstrings.",
        "Write production-ready Python code with type hints.",
    ]

    optimizer = EvoPromptOptimizer(
        task_desc="Python code generation",
        val_set=code_val_set,
        api_key="your-api-key-here",  # REPLACE
        api_base="https://api.x.ai/v1",
        model="grok-beta",
        population_size=20,
        num_generations=10,
        racing_threshold=3,
        seed_prompts=code_seed_prompts,
    )

    best_prompt = optimizer.evolve()
    return best_prompt


# ============================================================================
# Example 3: Text Classification
# ============================================================================

def example_text_classification():
    """Example: Optimizing prompts for sentiment analysis."""

    sentiment_val_set = [
        {
            'input': 'This movie was absolutely fantastic! I loved every minute.',
            'output': 'positive'
        },
        {
            'input': 'Terrible service, would not recommend to anyone.',
            'output': 'negative'
        },
        {
            'input': 'The product is okay, nothing special.',
            'output': 'neutral'
        },
        {
            'input': 'Best purchase I have ever made! Highly recommend!',
            'output': 'positive'
        },
        {
            'input': 'Complete waste of money. Very disappointed.',
            'output': 'negative'
        },
    ]

    optimizer = EvoPromptOptimizer(
        task_desc="sentiment classification",
        val_set=sentiment_val_set,
        api_key="your-api-key-here",  # REPLACE
        api_base="https://api.x.ai/v1",
        model="grok-beta",
        population_size=20,
        num_generations=10,
        racing_threshold=3,
    )

    best_prompt = optimizer.evolve()
    return best_prompt


# ============================================================================
# Example 4: Using with OpenAI instead of xAI
# ============================================================================

def example_openai_config():
    """Example: Using OpenAI GPT-4 instead of Grok."""

    math_val_set = [
        {
            'input': 'What is 15 + 27?',
            'output': '42'
        },
        {
            'input': 'What is 100 - 35?',
            'output': '65'
        },
        {
            'input': 'What is 12 * 8?',
            'output': '96'
        },
    ]

    optimizer = EvoPromptOptimizer(
        task_desc="basic arithmetic",
        val_set=math_val_set,
        api_key="sk-...",  # OpenAI API key
        api_base="https://api.openai.com/v1",  # OpenAI endpoint
        model="gpt-4",  # or "gpt-3.5-turbo"
        population_size=15,
        num_generations=8,
    )

    best_prompt = optimizer.evolve()
    return best_prompt


# ============================================================================
# Example 5: Custom Fitness Evaluation
# ============================================================================

def example_custom_fitness():
    """
    Example: How to extend the optimizer for custom fitness functions.

    This shows the framework's modularity - you can subclass and override
    _evaluate_prompt_on_example for custom evaluation logic.
    """

    class CustomEvoOptimizer(EvoPromptOptimizer):
        def _evaluate_prompt_on_example(self, prompt, example):
            """
            Custom evaluation that checks for exact match
            and also considers response length.
            """
            full_prompt = f"{prompt}\n\nProblem: {example['input']}\n\nAnswer:"

            try:
                response = self._call_llm(full_prompt, max_tokens=100, temperature=0.0)
                if not response:
                    return False

                # Exact match check
                response = response.strip().lower()
                expected = str(example['output']).strip().lower()

                # More strict: must be exact match, not just contains
                return response == expected

            except Exception as e:
                print(f"Error: {e}")
                return False

    val_set = [
        {'input': 'What is 2+2?', 'output': '4'},
        {'input': 'What is 3*3?', 'output': '9'},
    ]

    optimizer = CustomEvoOptimizer(
        task_desc="arithmetic",
        val_set=val_set,
        api_key="your-api-key-here",
        population_size=15,
        num_generations=8,
    )

    best_prompt = optimizer.evolve()
    return best_prompt


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Evolutionary Prompt Optimizer - Example Usage\n")
    print("=" * 60)
    print("Available examples:")
    print("1. Math Reasoning (GSM8K-style)")
    print("2. Code Generation")
    print("3. Text Classification")
    print("4. OpenAI Configuration")
    print("5. Custom Fitness Function")
    print("=" * 60)

    # Uncomment the example you want to run:

    # Example 1: Math reasoning
    # best_prompt = example_math_reasoning()

    # Example 2: Code generation
    # best_prompt = example_code_generation()

    # Example 3: Text classification
    # best_prompt = example_text_classification()

    # Example 4: OpenAI config
    # best_prompt = example_openai_config()

    # Example 5: Custom fitness
    # best_prompt = example_custom_fitness()

    print("\nTo run an example, uncomment the desired function call in __main__")
    print("Don't forget to replace 'your-api-key-here' with your actual API key!")
