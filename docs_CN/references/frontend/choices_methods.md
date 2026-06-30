<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# Choices Methods in SGLang
This doc describes the choices methods supported by SGLang.

The optional `choices_method` arg determines how options supplied to SGLang's `choices` primitive are selected. Only the `RuntimeEndpoint` backend supports the `choices_method` arg. Other backends, such as `OpenAI`, have bespoke selection implementations due to API limitations.

## Methods

### Token Length Normalized

Token length normalized is the default SGLang choices method. It selects the option with the highest average logprob across all of its tokens.

用法 example (alternatively, simply omit the `choices_method` arg):
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.token_length_normalized,
        )
    )
```


This can perform poorly if an option contains many tokens, where its later tokens are predicted with high confidence based on its earlier tokens. For instance, even strong models will fail the above example if the specified options are `["Paris", "Antidisestablishmentarianism"]`.

### Greedy Token Selection

Greedy token selection simply selects the option with the highest logprob for its initial token. For overlapping options where one option is a subset of a longer option, the logprobs of the shorter option are extended using its average logprob for comparison against the longer option.

用法 example:
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.greedy_token_selection,
        )
    )
```

This can perform poorly if an option misleads the model down a bad path based on an attractive initial token. For instance, greedy selection will result in an incorrect response for this example:
```python
@sgl.function
def us_president_example(s):
    s += sgl.user("Name a US president.")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["Donald Duck", "Millard Fillmore"],
            choices_method=sgl.greedy_token_selection,
        )
    )
```

### Unconditional Likelihood Normalized

Unconditional likelihood normalized selects the option with the highest average token logprob once normalized by the unconditional token logprobs, as described in [this EleutherAI blogpost](https://blog.eleuther.ai/multiple-choice-normalization/). This method incurs an additional LLM call to obtain the unconditional likelihoods.

用法 example:
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.unconditional_likelihood_normalized,
        )
    )
```
