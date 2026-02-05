"""Subpackage for the loss function.
"""

def is_logits(data): 
    """Attempts to guess whether the given PyTorch tensor contains logits.

    If the argument `data` contains only values that are no less than 0 and no
    greater than 1, then `False` is returned; otherwise, `True` is returned.
    """
    if   (data > 1).any(): return True
    elif (data < 0).any(): return True
    else:                  return False
# writing a test DICE loss coefficient        
def dice_loss(pred, gold, logits=None, smoothing=0, metrics=None):
    """Returns the loss based on the dice coefficient.
    
    `dice_loss(pred, gold)` returns the dice-coefficient loss between the
    tensors `pred` and `gold` which must be the same shape and which should
    represent probabilities. The first two dimensions of both `pred` and `gold`
    must represent the batch-size and the classes.

    Parameters
    ----------
    pred : tensor
        The predicted probabilities of each class.
    gold : tensor
        The gold-standard labels for each class.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    smoothing : number, optional
        The smoothing coefficient `s`. The default is `1`.
    metrics : dict or None, optional
        An optional dictionary into which the key `'dice'` should be inserted
        with the dice-loss as the value.

    Returns
    -------
    float
        The dice-coefficient loss of the prediction.
    """
    pred = pred.contiguous()
    gold = gold.contiguous()
    if logits is None: logits = is_logits(pred) #sometimes logit, sometimes probability, this func automatically detect whether logit or prob
    if logits: pred = torch.sigmoid(pred)
    intersection = (pred * gold) #high probabilities get higher values, low get low, gold is 0s and 1s, this gives predicted probabilities where true value is correct
    pred = pred**2 #noah checking if we should be squaring here
    gold = gold**2
    while len(intersection.shape) > 2:
        intersection = intersection.sum(dim=-1)
        pred = pred.sum(dim=-1)
        gold = gold.sum(dim=-1)
    if smoothing is None: smoothing = 0
    loss = (1 - ((2 * intersection + smoothing) / (pred + gold + smoothing)))
    # Average the loss across classes then take the mean across batch elements.
    loss = loss.mean(dim=1).mean() #utilities that we can ignore - mean across channels and then across all the batches
    if metrics is not None:
        if 'dice' not in metrics: metrics['dice'] = 0.0
        metrics['dice'] += loss.data.cpu().numpy() * gold.size(0)
    return loss
