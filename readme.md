Hello everyone, this is something I am currently doing. This here is a Naive implementation of KV cache done on top of Qwen 2.5 0.5B Instuctiion Tuned. It is a static cache. Takes only one prompt as input, and max new tokens need to be specified. No batching and other things.

<br>
So instead of model.forward(), you can use a .generate function built in the model class that takes use_cache = bool as a parameter as in the models on Huggingface. So, inference is possible.