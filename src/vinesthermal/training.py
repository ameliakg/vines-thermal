"""The code for training the model.

This is the code that was used to train the original vines-thermal model.
"""

import torch

from .loss import dice_loss

def train_model(lr, gamma, batch_size, n_epochs=10, base_model='resnet18', print=None):
    """TODO: write docstring. See the `dice_loss` function in `loss.py` for an example.
    Typically we try to keep them 79 characters wide or less. 
    """
    if print is None: 
        print = lambda *args,**kw: None
    print("Training Model...")
    print(f"  lr    = {lr}")
    print(f"  gamma = {gamma}")
    print(f"  bsize = {batch_size}")
    
    model = UNet(base_model=base_model)
    
    # Make the optimizer and LR-manager:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #optimizer handles updating the parameters each step
    steplr = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=gamma) #take smaller steps in the learning rate as you get closer
    
    # Declare our loss function: what's actually getting minimized
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()  #loss function that works for pixels and logits- well established
    dice_loss_fn = lambda a, b: dice_loss(a, b, smoothing=0, logits=True)
    both_loss_fn = lambda a,b,w=0.5: (1-w)*bce_loss_fn(a,b) + w*dice_loss_fn(a,b)
    
    
    # Make the dataloaders:
    train_dloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True)
    test_dloader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=len(test_ds),
        shuffle=False)
    train_eval_dloader = torch.utils.data.DataLoader(
        train_eval_ds,
        batch_size=len(train_eval_ds),
        shuffle=False)
    
    # Now we start the optimization loop:
    for epoch_num in range(n_epochs):
        print(f"    * Epoch: {epoch_num}")
        #loss_fn = lambda a,b: both_loss_fn(a, b, w=(epoch_num + 1)/ n_epochs)
        loss_fn = both_loss_fn
        # Put the model in train mode:
        model.train()
        # In each epoch, we go through each training sample once; the dataloader
        # gives these to us in batches:
        total_train_loss = 0
        for (inputs, targets) in train_dloader:
            # We're starting a new step, so we reset the gradients.
            optimizer.zero_grad()
            # Calculate the model prediction for these inputs.
            preds = model(inputs)
            # Calculate the loss between the prediction and the actual outputs.
            train_loss = loss_fn(preds, targets) #sigmoid gives the probability but don't need sigmoid bc of BCE w/logit loss
            # Have PyTorch backward-propagate the gradients.
            train_loss.backward()
            # Have the optimizer take a step: (update the parameters)
            optimizer.step()
            # Add up the total training loss:
            total_train_loss += float(train_loss.detach())*len(targets)
            train_loss = None
        # LR Scheduler step:
        steplr.step() #make the learning rate smaller
        mean_train_loss = total_train_loss / len(train_ds)
        if not np.isfinite(mean_train_loss):
            return (model, np.nan)
        # Now that we've finished training, put the model back in evaluation mode.
        with torch.no_grad():
            model.eval()
            ## Evaluate the model using the test data.
            total_test_dice_loss = 0
            total_test_bce_loss = 0
            total_test_loss = 0
            for (inputs, targets) in test_dloader:
                preds = model(inputs)
                test_loss = loss_fn(preds, targets)
                total_test_loss += float(test_loss.detach()) * len(targets) # changed from train loss
                total_test_dice_loss += float(dice_loss_fn(preds, targets).detach()) * len(targets)
                total_test_bce_loss += float(bce_loss_fn(preds, targets).detach()) * len(targets)
            mean_test_loss = total_test_loss / len(test_ds)
            mean_test_dice_loss = total_test_dice_loss / len(test_ds)
            mean_test_bce_loss = total_test_bce_loss / len(test_ds)
            total_train_dice_loss = 0
            total_train_bce_loss = 0
            for (inputs, targets) in train_eval_dloader:
                preds = model(inputs)
                total_train_dice_loss += float(dice_loss_fn(preds, targets).detach()) * len(targets)
                total_train_bce_loss += float(bce_loss_fn(preds, targets).detach()) * len(targets)
            mean_train_dice_loss = total_train_dice_loss / len(train_eval_ds)
            mean_train_bce_loss = total_train_bce_loss / len(train_eval_ds)
        # Print something about this step:
        print(
            f"      train loss: "
            f"{mean_train_loss:6.3f} [{mean_train_dice_loss:6.3f} {mean_train_bce_loss:6.3f}]")
        print(
            f"      test loss: "
            f"{mean_test_loss:6.3f} [{mean_test_dice_loss:6.3f} {mean_test_bce_loss:6.3f}]")
    # After the optimizer has run, print out what it's found:
    print("Final result:")
    print(f"  train dice loss = ", float(mean_train_dice_loss))
    print(f"  test dice loss = ", float(mean_test_dice_loss))
    return (model, float(mean_test_dice_loss))
