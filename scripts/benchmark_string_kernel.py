import random
import timeit
from sklearn_jax_kernels.structured.string_utils import (
    AsciiBytesTransformer, NGramTransformer)
from sklearn_jax_kernels.structured.strings import SpectrumKernel
from sklearn.pipeline import Pipeline

strings = [
    "".join(random.choices(['A', 'T', 'G', 'C'], k=115)) for i in range(1000)]
pipeline = Pipeline([
    ('bytes', AsciiBytesTransformer()),
    ('ngrams', NGramTransformer(2))
])
transformed = pipeline.transform(strings).block_until_ready()
kernel = SpectrumKernel(n_gram_length=None)

kernel(transformed).block_until_ready()
print('Compiled')
time = timeit.timeit(lambda: kernel(transformed).block_until_ready(), number=5)
print(time)

