# Theory recap

## Tokens

A tokenizer converts text into integer IDs from its vocabulary. Tokens can represent words, parts of words, punctuation, or other units. Special tokens can mark conversation roles or boundaries. The exact IDs depend on the tokenizer; we will inspect actual values instead of inventing them.

The model uses IDs to look up learned embedding vectors. An ID is an index, not a semantic score. We explore tokenization and embeddings in the next topics.

## Learned weights

Weights are numerical parameters learned during training, including embedding tables and transformation matrices. Different inputs produce different intermediate values even when weights stay fixed. For example, the fixed rule `y = 2x` gives different outputs for different `x`.

In ordinary inference, the model does not learn by changing these weights. Prompt context and temporary computation change. A later KV-cache lesson will explain another changing runtime state that is distinct from weights.

## Forward pass

A forward pass evaluates the model on its inputs. A causal language model produces logits over its vocabulary at input positions. For a single sequence, the full logits tensor has shape `[1, input_length, vocabulary_size]`. The final position's scores are used to choose the next token. Greedy selection chooses the largest score; sampling uses a probability distribution instead.

Generation repeats model computation as new tokens become part of the context. It is more than one forward pass. With KV caching, later steps can reuse previously computed keys and values rather than recomputing all prior positions.

## Training and inference

| Stage | Typical steps | Weight update? |
|---|---|---|
| Training | Forward pass, loss, backward pass, optimizer step | The optimizer updates trainable parameters |
| Inference | Forward pass and output selection | No ordinary weight update |

`model.eval()` changes training-sensitive layer behavior such as dropout. It does not itself freeze parameters. `torch.inference_mode()` disables autograd recording and related overhead in this lab. There is no backward call or optimizer here.

## Model cards

Before use, inspect the model identity/revision, base versus instruction-tuned purpose, architecture, tokenizer/chat formatting, license, intended uses, limitations, and reported evaluations. Reported benchmark scores are not a guarantee for our prompts. Use the [model-card worksheet](model-card.md).

## CPU and GPU

A CPU has general-purpose cores and vector instructions; optimized math libraries already execute parallel arithmetic. A GPU is designed for high-throughput parallel work and uses a different execution and memory organization. Speed depends on workload, memory transfers, batching, and kernels; a GPU is not automatically faster for every tiny operation.

Your local CPU is an i7-1355U with 10 physical cores and 12 logical processors. The integrated Iris Xe shares system memory. Our GPU practical will use the CUDA GPU actually allocated by Colab. More Python workers cannot turn the CPU into GPU hardware; the research question is how much useful overlap and parallelism software can expose.

## References

- [Selected teaching model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [Transformers model interface](https://huggingface.co/docs/transformers/en/main_classes/model)
- [PyTorch autograd introduction](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

No video is assigned: the learner correctly explained that weights stay fixed during ordinary inference.
