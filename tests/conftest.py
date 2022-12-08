from hypothesis import settings

settings.register_profile("dev", deadline=None, max_examples=100)
settings.register_profile("dev_cpu", deadline=None, max_examples=100)
settings.register_profile("dev_gpu", deadline=None, max_examples=1000)

settings.register_profile("ci", deadline=None, max_examples=1000)
settings.register_profile("ci_cpu", deadline=None, max_examples=1000)
settings.register_profile("ci_gpu", deadline=None, max_examples=10000)

settings.register_profile("ci_slow", deadline=None, max_examples=10000)
settings.register_profile("ci_cpu_slow", deadline=None, max_examples=10000)
settings.register_profile("ci_gpu_slow", deadline=None, max_examples=100000)
