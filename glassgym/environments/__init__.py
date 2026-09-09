__all__ = ["GreenLightEnv"]


def __getattr__(name):
    if name == "GreenLightEnv":
        from glassgym.environments.greenlight_env import GreenLightEnv

        return GreenLightEnv
    raise AttributeError(name)
