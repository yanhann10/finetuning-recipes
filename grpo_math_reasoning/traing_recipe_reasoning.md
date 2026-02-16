# Training recipe and learnings

## Reward shaping

### Reward scaling

Finding the right ratio between rewards matters. In this project I used correctness reward of 1.0 and formatting reward of 0.1.

- Previously the formatting reward was higher, which resulted in reward hacking by generating perfect tags without meaningful content.
- Large correctness reward value could increase variance in group advantage and could make training unstable.
- Adding arbitrary rewards, such as a length penalty, did not improve performance in this project.

## Partial credit

- Distance-based partial credit given to answer similar to ground truth answer had minimal effect in this project. In math reasoning tasks, near-miss answers usually indicate logical failure rather than minor rounding errors. Binary rewards provide a cleaner signals.
- As there could be different reasoning paths to solve a math problem, either a model-based approach or a code-execution based approach which turns maths into code would be helpful as step-wise credit.

## KL divergence

A small KL penalty is useful.

- Helps keep reasoning grounded in the base model’s prior distribution.
