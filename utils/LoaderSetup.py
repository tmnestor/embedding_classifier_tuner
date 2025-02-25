import yaml


def join_constructor(loader, node):
    # Joins the list of strings into one string.
    seq = loader.construct_sequence(node)
    return "".join(seq)


# Register the constructor for the !join tag with the SafeLoader.
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
