from omegaconf import OmegaConf

from iht.utils.misc import import_fn

# Resolvers used in hydra configs
# (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied
#  in multiple places in the config.
# used primarily for num_ensv
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)
# to convert things like -z, +y to minus_z, y, which is better as save path
OmegaConf.register_new_resolver(
    "pretty_axis", lambda s: f"minus_{s[1]}" if s[0] == "-" else s[1]
)
# imports an object
OmegaConf.register_new_resolver("import", lambda s: import_fn(s))
