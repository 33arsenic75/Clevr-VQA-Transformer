"""utils.py: Utility functions for training, evaluation, visualization, and saving/loading models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
import numpy as np
from torchvision import transforms # For unnormalizing image in visualize_preds


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multi-class classification tasks."""
    def __init__(self, gamma=2.0, weight=None, ignore_index=-1):
        """
        Args:
            gamma (float): Focusing parameter. Default is 2.0.
            weight (Tensor, optional): A manual rescaling weight given to each class.
                                       If given, it has to be a Tensor of size [C].
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """Forward pass for Focal Loss.
        Args:
            inputs (Tensor): Predictions from the model (logits).
            targets (Tensor): True labels.
        """
        """Apply softmax to get probabilities"""
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        """Gather log probabilities at true class indices"""
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * logpt

        """Ignore entries where target == ignore_index"""
        valid_mask = targets != self.ignore_index
        return loss[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, requires_grad=True, device=inputs.device)


def evaluate_model(model, dataloader, criterion, device, full_report=False):
    """Evaluate the model on the validation/test set.
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation/test set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform evaluation on.
        full_report (bool): If True, print classification report.
    Returns:
        avg_loss (float): Average loss over the dataset.
        accuracy (float): Accuracy of the model on the dataset.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total_valid_samples = 0 # Samples not ignored by loss
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, questions, attn_masks, targets in dataloader:
            # Move data to the specified device
            images = images.to(device)
            questions = questions.to(device)
            attn_masks = attn_masks.to(device)
            targets = targets.to(device) #
            
            """print(f"Batch {i}: {(targets != -1).sum().item()} valid targets")"""

            """Skip batch if all targets are -1 (ignore_index)"""
            if torch.all(targets == -1):
                continue

            outputs = model(images, questions, attn_masks) #
            
            """Calculate loss only on valid targets"""
            loss = criterion(outputs, targets) # criterion handles ignore_index
            
            _, predicted = outputs.max(1) #

            """For metrics, consider only where targets are not ignore_index"""
            valid_targets_mask = (targets != -1)
            num_valid_in_batch = valid_targets_mask.sum().item()

            if num_valid_in_batch > 0:
                """Loss item is average over batch, scale by num_valid_in_batch if criterion reduction is 'mean'
                If reduction is 'sum', then loss.item() is already sum. Assuming 'mean'."""
                running_loss += loss.item() * num_valid_in_batch 
                correct += predicted[valid_targets_mask].eq(targets[valid_targets_mask]).sum().item() #
                total_valid_samples += num_valid_in_batch
            
            all_preds.extend(predicted.cpu().numpy()) #
            all_labels.extend(targets.cpu().numpy()) #

    avg_loss = running_loss / total_valid_samples if total_valid_samples > 0 else 0
    accuracy = correct / total_valid_samples if total_valid_samples > 0 else 0 #

    if full_report: #
        print("\nClassification Report:") #
        """Filter out ignored labels (-1) for classification report"""
        report_labels = [label for label in all_labels if label != -1]
        report_preds = [pred for i, pred in enumerate(all_preds) if all_labels[i] != -1]
        if report_labels and report_preds:
            print(classification_report(report_labels, report_preds, zero_division=0)) #
        else:
            print("Not enough valid samples to generate a classification report.")
            
    return avg_loss, accuracy


def plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir="."):
    """Plot training and validation loss and accuracy curves.
    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_accs (list): List of training accuracies.
        val_accs (list): List of validation accuracies.
        out_dir (str): Directory to save the plots.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))


    """Plot training and validation loss"""
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    """Plot training and validation accuracy"""
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    """ Adjust layout to prevent overlap """
    plt.tight_layout()

    save_file = os.path.join(out_dir, "training_curves.png")
    plt.savefig(save_file)
    print(f"Training curves saved to {save_file}")
    plt.close()


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    """Save the model and optimizer state.
    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): Current epoch number.
        path (str): Path to save the checkpoint.
    """
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    } #
    torch.save(state, path) #


def load_checkpoint(model, path="checkpoint.pth"):
    """Ensure loading onto CPU first, then model can be moved to device"""
    state = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state['model_state_dict'])
    """Optimizer state and epoch can also be returned if needed for resuming training"""
    return model


def visualize_preds(model, dataset, tokenizer, device, out_dir=".", correct=True, count=5, title="Predictions"):
    import random
    model.eval()
    """Ensure model is in evaluation mode"""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    """Randomly select samples to visualize"""
    """Shuffle indices to ensure randomness"""
    """Note: This is a simple random selection. For more complex sampling, consider stratified sampling."""
    
    indices_to_show = []
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices) #

    """Select a subset of indices to visualize"""
    for idx in all_indices:
        if len(indices_to_show) >= count:
            break
        
        """Unpack dataset item"""
        image_tensor, question_ids, attn_mask, label_idx_tensor = dataset[idx] #
        label_idx = label_idx_tensor.item()

        """Skip if label is -1 (OOV) and we are looking for correct predictions"""
        if label_idx == -1 and correct:
            continue
        """Skip if label is -1 (OOV) and we are looking for incorrect, as "incorrect" isn't well-defined"""
        if label_idx == -1 and not correct:
             continue

        """Check if prediction is correct"""
        image_input = image_tensor.unsqueeze(0).to(device) #
        question_input = question_ids.unsqueeze(0).to(device) #
        attn_mask_input = attn_mask.unsqueeze(0).to(device) #

        """Ensure attention mask is correct size"""
        with torch.no_grad(): #
            output = model(image_input, question_input, attn_mask_input) #
            pred_idx = output.argmax(1).item() #

        is_prediction_correct = (pred_idx == label_idx)

        if (is_prediction_correct and correct) or (not is_prediction_correct and not correct):
            indices_to_show.append(idx)

    if not indices_to_show:
        print(f"No samples found for visualization: {title}")
        return

    """Ensure count does not exceed found samples"""
    actual_count = min(count, len(indices_to_show))
    fig, axes = plt.subplots(1, actual_count, figsize=(actual_count * 4, 5))
    if actual_count == 1:
        axes = [axes] # Make it iterable

    for i, sample_idx in enumerate(indices_to_show[:actual_count]):
        image_tensor, question_ids, _, label_idx_tensor = dataset[sample_idx]
        label_idx = label_idx_tensor.item()

        """Re-predict for safety, though it should be the same"""
        image_input = image_tensor.unsqueeze(0).to(device)  

        question_input = question_ids.unsqueeze(0).to(device)
        attn_mask_input = torch.ones_like(question_ids).unsqueeze(0).to(device)
                                                                               
        """Ensure attention mask is correct size"""
        _, _, attn_mask_orig, _ = dataset[sample_idx]
        attn_mask_input = attn_mask_orig.unsqueeze(0).to(device)

        """Ensure model is in evaluation mode"""
        with torch.no_grad():
            output = model(image_input, question_input, attn_mask_input)
            pred_idx = output.argmax(1).item()

        """Decode question and answers"""
        question_text = tokenizer.decode(question_ids, skip_special_tokens=True) #
        gt_ans = dataset.decode_answer(label_idx) if label_idx != -1 else "N/A (OOV)" #
        pred_ans = dataset.decode_answer(pred_idx) #

        """Unnormalize image for display"""
        """Assuming image_tensor is a CHW tensor from the dataset"""
        img_to_display = image_tensor.cpu().clone() # Work on a CPU copy
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        """Unnormalize the image tensor"""
        img_to_display = inv_normalize(img_to_display)
        img_to_display = transforms.ToPILImage()(img_to_display)

        """Convert to numpy array for display"""
        ax = axes[i]
        ax.imshow(img_to_display) #
        ax.set_title(f"Q: {question_text}\nPred: {pred_ans} | GT: {gt_ans}", fontsize=8) #
        ax.axis('off') #

        """Add prediction correctness indicator"""

    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    """ Save the figure with a safe filename """   
    filename_safe_title = "".join(c if c.isalnum() else "_" for c in title)
    save_file = os.path.join(out_dir, f"{filename_safe_title}.png")
    plt.savefig(save_file)
    print(f"Visualization saved to {save_file}")
    plt.close() # Close the figure to free memory