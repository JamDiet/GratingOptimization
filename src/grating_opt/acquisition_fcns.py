from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.adapter.registry import Generators

# Import the specific BoTorch acquisition function class
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

def ucb(seed_trials: int = 10, kappa: float = 2.0):
    return GenerationStrategy(
        name="Sobol+UCB",
        nodes=[
            GenerationNode(
                name="Sobol",
                generator_specs=[
                    GeneratorSpec(
                        generator_enum=Generators.SOBOL
                    )
                ],
                transition_criteria=[
                    MinTrials(
                        # Transition to the 'UCB' node once seed_trials are complete
                        threshold=seed_trials,
                        transition_to="UCB",
                        use_all_trials_in_exp=True
                    )
                ],
            ),
            GenerationNode(
                name="UCB",
                generator_specs=[
                    GeneratorSpec(
                        generator_enum=Generators.BOTORCH_MODULAR,
                        # Pass the acquisition function configuration here
                        model_kwargs={
                            "botorch_acqf_class": qUpperConfidenceBound,
                            "acquisition_options": {
                                "beta": kappa  # 'beta' maps to your kappa exploration parameter
                            }
                        }
                    )
                ]
            )
        ]
    )