from hypothesis import settings

# from pytest import Parser, Metafunc, Config


# def pytest_addoption(parser: Parser) -> None:
#     parser.addoption("--gpu_only", action="store_true", help="run gpu only")
#     parser.addoption("--cpu_only", action="store_true", help="run cpu only")
#     parser.addoption("--max_examples", type=int, default=1000, help="max examples")


# def pytest_configure(config: Config) -> None:
#     pass


settings.register_profile("dev", deadline=None, max_examples=100)
settings.register_profile("dev_cpu", deadline=None, max_examples=100)
settings.register_profile("dev_gpu", deadline=None, max_examples=100)

settings.register_profile("ci", deadline=None, max_examples=1000)
settings.register_profile("ci_cpu", deadline=None, max_examples=1000)
settings.register_profile("ci_gpu", deadline=None, max_examples=1000)

settings.register_profile("ci_slow", deadline=None, max_examples=10000)
settings.register_profile("ci_cpu_slow", deadline=None, max_examples=10000)
settings.register_profile("ci_gpu_slow", deadline=None, max_examples=10000)
