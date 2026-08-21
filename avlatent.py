"""Reaching inside H3's paired (video, audio) latent.

One helper, in its own module because the video, masking and long-form nodes all
need it and none of them should have to import each other to get it.
"""


def av(samples):
    """(video, audio) out of an H3 AV latent.

    NestedTensor.__getitem__ BROADCASTS the index into every contained tensor
    rather than selecting one, so `samples[0]` / `samples[1]` silently does the
    wrong thing and then IndexErrors. The tensors are reached via `.tensors`.
    """
    t = getattr(samples, "tensors", None)
    if t is None:
        t = samples.unbind() if hasattr(samples, "unbind") else samples
    return t[0], t[1]
