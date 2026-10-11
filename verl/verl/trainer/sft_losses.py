"""Plain-probability SFT objective for the single-turn FSDP trainer.

The detached target-token probability rescales the NLL gradient. This implements
the gradient of -p without materializing a second [tokens, vocabulary] softmax tensor.
"""


def plain_probability_loss(token_nll):
    """Return a differentiable surrogate whose gradient equals that of -p.

    The scalar value is -p + a constant/term with zero gradient only in the
    gradient sense; report the detached probability if the true value is needed.
    """
    return token_nll * (-token_nll.detach()).exp()
