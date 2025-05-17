
"""part1.py: Implements training and evaluation for Parts 8, 9, 10"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import CLEVRVQADataset
from model import VQAModel
from utils import evaluate_model, plot_curves, visualize_preds, load_checkpoint, save_checkpoint, FocalLoss

BATCH_SIZE = 512 # From original file
MAX_Q_LEN = 30 # Default from model.py and dataset.py constructor
NUM_EPOCHS = 10

def train(args):
    """Set device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    """Set up save path"""
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        print(f"Created directory for saving outputs: {args.save_path}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    """Create training dataset - this will build the vocabulary"""
    print("Loading training data (trainA) and building vocabulary...")
    train_data = CLEVRVQADataset(args.dataset, split="trainA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN) #

    """Share vocabulary from train_data"""
    shared_answer_to_idx = train_data.answer_to_idx
    shared_idx_to_answer = train_data.idx_to_answer
    shared_num_answers = train_data.num_answers
    print(f"Vocabulary built: {shared_num_answers} unique answers found in training data.")

    print("Loading validation data (valA)...")
    val_data = CLEVRVQADataset(args.dataset, split="valA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN,
                               answer_to_idx=shared_answer_to_idx,
                               idx_to_answer=shared_idx_to_answer,
                               precomputed_num_answers=shared_num_answers)

    """Check dataset sizes"""
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")

    """Create DataLoader for training and validation"""
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2) #
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True) #

    """Check DataLoader sizes"""
    print(f"Torch DataLoader: {len(train_loader)} training batches, {len(val_loader)} validation batches.") #

    """Initialize model"""
    model = VQAModel(vocab_size=len(tokenizer), num_classes=shared_num_answers, max_len=MAX_Q_LEN) #
    model = model.to(device)
    
    """Load pretrained weights if available"""
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay = 1e-4) #
    """If you used target=-1 for OOV answers, CrossEntropyLoss should ignore it."""
    criterion = nn.CrossEntropyLoss(ignore_index=-1) #

    print("Setting up training...") #
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], [] #

    num_epochs = NUM_EPOCHS # As in the original code
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        """Training loop"""
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples_epoch = 0


        for batch_idx, batch in enumerate(train_loader):
            images, questions, attn_masks, targets = batch
            
            images = images.to(device)
            questions = questions.to(device)
            attn_masks = attn_masks.to(device)
            targets = targets.to(device)
            
            """Skip batch if all targets are -1 (ignore_index) after potential filtering"""
            if torch.all(targets == -1):
                print(f"Skipping batch {batch_idx+1} in epoch {epoch+1} as all targets are to be ignored.")
                continue

            """Move data to device"""
            optimizer.zero_grad()
            outputs = model(images, questions, attn_masks)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(dim=1)
            
            """Consider only valid targets for accuracy calculation (not -1)"""
            valid_targets_mask = targets != -1
            num_valid_targets_in_batch = valid_targets_mask.sum().item()

            """Accumulate loss and accuracy only for valid targets"""
            # Note: This is a correction from the original code where total_samples_epoch was not updated correctly
            if num_valid_targets_in_batch > 0:
                running_loss += loss.item() * num_valid_targets_in_batch # Accumulate loss scaled by valid samples
                running_correct += predicted[valid_targets_mask].eq(targets[valid_targets_mask]).sum().item()
                total_samples_epoch += num_valid_targets_in_batch
            
            """Print progress every 50 batches"""
            if batch_idx % 50 == 0 and num_valid_targets_in_batch > 0: # Print progress less frequently
                batch_loss = loss.item()
                batch_acc = predicted[valid_targets_mask].eq(targets[valid_targets_mask]).sum().item() / num_valid_targets_in_batch
                print(f"[Epoch {epoch+1}/{num_epochs}][Batch {batch_idx+1}/{len(train_loader)}] "
                      f"Batch Loss: {batch_loss:.4f}, Batch Acc: {batch_acc:.4f}")


        """End of epoch: calculate average loss and accuracy"""
        train_loss = running_loss / total_samples_epoch if total_samples_epoch > 0 else 0
        train_acc = running_correct / total_samples_epoch if total_samples_epoch > 0 else 0 # CORRECTED
        train_losses.append(train_loss)
        train_accs.append(train_acc)


        """Validation loop"""
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device) # Pass device
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")


        """Save model if validation accuracy improved"""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.save_path, "best_model_epoch.pth")
            save_checkpoint(model, optimizer, epoch, path=checkpoint_path) #
            print(f"New best model saved with Val Acc: {best_val_acc:.4f} at {checkpoint_path}")
    
    print("Training finished.")
    plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir=args.save_path) #
    print(f"Training curves saved to {args.save_path}")

    print("\nLoading best model for final evaluation and visualization...")
    best_model_path = os.path.join(args.save_path, "best_model_epoch.pth")
    if os.path.exists(best_model_path):
        model = load_checkpoint(model, best_model_path) #
        model = model.to(device) # Ensure model is on correct device after loading
    else:
        print(f"Warning: Best model checkpoint '{best_model_path}' not found. Using model from last epoch.")


    print("Loading test data (testA)...")
    test_data = CLEVRVQADataset(args.dataset, split="testA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN,
                                answer_to_idx=shared_answer_to_idx,
                                idx_to_answer=shared_idx_to_answer,
                                precomputed_num_answers=shared_num_answers) #
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE) #

    print("\nEvaluating on Test Set (testA):")
    evaluate_model(model, test_loader, criterion, device, full_report=True) #
    
    print("\nVisualizing predictions on Test Set (testA):")
    visualize_preds(model, test_data, tokenizer, device, out_dir=args.save_path, correct=True, count=5, title="Correct_Predictions_testA") #
    visualize_preds(model, test_data, tokenizer, device, out_dir=args.save_path, correct=False, count=5, title="Incorrect_Predictions_testA") #

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for inference.")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #

    """For inference, we need to know the number of classes the model was trained with.
    We re-create the vocabulary based on 'trainA' split from the dataset path."""
    print("Building vocabulary reference from 'trainA' for consistent model loading...")
    train_ref_dataset = CLEVRVQADataset(args.dataset, split="trainA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN)
    num_classes = train_ref_dataset.num_answers
    shared_answer_to_idx = train_ref_dataset.answer_to_idx
    shared_idx_to_answer = train_ref_dataset.idx_to_answer
    print(f"Model is expected to have {num_classes} output classes.")

    """Assuming inference is on "testA" for Part 1. This could be an argument later."""
    eval_split = "testA" 
    print(f"Loading inference data ({eval_split})...")
    test_data = CLEVRVQADataset(args.dataset, split=eval_split, tokenizer=tokenizer, max_q_len=MAX_Q_LEN,
                                answer_to_idx=shared_answer_to_idx,
                                idx_to_answer=shared_idx_to_answer,
                                precomputed_num_answers=num_classes) #
    test_loader = DataLoader(test_data, batch_size=1024) #

    """Load model and checkpoint"""
    model = VQAModel(vocab_size=len(tokenizer), num_classes=num_classes, max_len=MAX_Q_LEN) #
    print(f"Loading model from: {args.model_path}")
    
    """Load the checkpoint. This should be the best model from training."""
    ckpt_path = os.path.join(args.model_path, "best_model_epoch_beforeFinetune.pth") 
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    model = load_checkpoint(model, ckpt_path)
    
    """Move model to device"""
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=-1) # Use ignore_index if OOV answers are -1
    print(f"\nPerforming inference on '{eval_split}' split:")
    evaluate_model(model, test_loader, criterion, device, full_report=True) #
    
def finetune_image_encoder(args):
    """Fine-tune the image encoder (ResNet) of the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    """— rebuild shared vocab from trainA —"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("Loading trainA to rebuild vocabulary...")
    train_data = CLEVRVQADataset(
        args.dataset, split="trainA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN
    )
    shared_answer_to_idx = train_data.answer_to_idx
    shared_idx_to_answer = train_data.idx_to_answer
    num_classes = train_data.num_answers

    """— load trainA and valA data with shared vocab —"""
    val_data = CLEVRVQADataset(
        args.dataset, split="valA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN,
        answer_to_idx=shared_answer_to_idx,
        idx_to_answer=shared_idx_to_answer,
        precomputed_num_answers=num_classes
    )

    """— create DataLoader for trainA and valA —"""
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader   = DataLoader(val_data,   batch_size=64,
                              num_workers=4, pin_memory=True)

    """— load model & checkpoint —"""
    print("Initializing model and loading best frozen checkpoint...")
    model = VQAModel(vocab_size=len(tokenizer),
                     num_classes=num_classes,
                     max_len=MAX_Q_LEN)
    ckpt_path = os.path.join(args.pretrained_model_path, "best_model_epoch_beforeFinetune.pth") 
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    model = load_checkpoint(model, ckpt_path)
    model = model.to(device)

    """— unfreeze ResNet —"""
    print("Unfreezing ResNet encoder parameters...")
    for p in model.resnet.parameters():
        p.requires_grad = True

    """— optimizer, loss —"""
    ft_lr = 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    """— tracking lists —"""
    FT_EPOCHS = 3
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0

    """— fine-tuning loop —"""
    print(f"Starting fine-tuning for {FT_EPOCHS} epochs (LR={ft_lr})…")
    for epoch in range(FT_EPOCHS):
        
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        batch_idx = 0
        for batch in train_loader:
            batch_idx += 1
            images, questions, attn_masks, targets = [x.to(device) for x in batch]
            if torch.all(targets == -1):
                continue

            """ skip batch if all targets are -1 (ignore_index) after potential filtering"""
            optimizer.zero_grad()
            outputs = model(images, questions, attn_masks)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            """ consider only valid targets for accuracy calculation (not -1) """
            mask = (targets != -1)
            n_valid = mask.sum().item()
            running_loss    += loss.item() * n_valid
            running_correct += (outputs.argmax(1)[mask] == targets[mask]).sum().item()
            total_samples   += n_valid
            
            """ print progress every 50 batches """
            if batch_idx % 50 == 0:
                batch_loss = loss.item()
                batch_acc = (outputs.argmax(1)[mask] == targets[mask]).sum().item() / n_valid
                print(f"[FT Epoch {epoch+1}/{FT_EPOCHS}][Batch {batch_idx}/{len(train_loader)}] "
                      f"Batch Loss: {batch_loss:.4f}, Batch Acc: {batch_acc:.4f}")
                

        """ end of epoch: calculate average loss and accuracy """
        train_loss = running_loss / total_samples
        train_acc  = running_correct / total_samples
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        """ validation loop """
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"[FT Epoch {epoch+1}/{FT_EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        "save if improved"
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path = os.path.join(args.save_path, "finetuned_best.pth")
            save_checkpoint(model, optimizer, epoch, out_path)
            print(f"→ New best fine-tuned model saved at epoch {epoch+1}: {out_path}")

    """— after fine-tuning: plot curves —"""
    print("Fine-tuning complete. Plotting curves…")
    plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir=args.save_path)

    """— final evaluation & visualization —"""
    print("\nLoading best fine-tuned model for final evaluation and visualization…")
    best_ft = os.path.join(args.save_path, "finetuned_best.pth")
    model = load_checkpoint(model, best_ft).to(device)

    print("Loading test data (testA)…")
    test_data = CLEVRVQADataset(
        args.dataset, split="testA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN,
        answer_to_idx=shared_answer_to_idx,
        idx_to_answer=shared_idx_to_answer,
        precomputed_num_answers=num_classes
    )
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    print("\nEvaluating on Test Set (testA):")
    evaluate_model(model, test_loader, criterion, device, full_report=True)

    print("\nVisualizing predictions on Test Set (testA):")
    visualize_preds(model, test_data, tokenizer, device,
                    out_dir=args.save_path, correct=True,
                    count=5, title="Correct_Preds_FineTuned")
    visualize_preds(model, test_data, tokenizer, device,
                    out_dir=args.save_path, correct=False,
                    count=5, title="Incorrect_Preds_FineTuned")

def train_with_focal_loss(args):
    """Fine-tune the image encoder (ResNet) of the model with Focal Loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("Loading trainA and valA data with shared vocab...")

    """— rebuild shared vocab from trainA —"""
    train_data = CLEVRVQADataset(args.dataset, split="trainA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN)
    shared_answer_to_idx = train_data.answer_to_idx
    shared_idx_to_answer = train_data.idx_to_answer
    num_classes = train_data.num_answers

    """— load trainA and valA data with shared vocab —"""
    val_data = CLEVRVQADataset(args.dataset, split="valA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN,
                                answer_to_idx=shared_answer_to_idx,
                                idx_to_answer=shared_idx_to_answer,
                                precomputed_num_answers=num_classes)

    """— create DataLoader for trainA and valA —"""
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_data, batch_size=64, num_workers=2, pin_memory=True)

    """— load model & checkpoint —"""
    print("Initializing model and loading best fine-tuned weights...")
    model = VQAModel(vocab_size=len(tokenizer), num_classes=num_classes, max_len=MAX_Q_LEN)
    ckpt_path = os.path.join(args.pretrained_model_path, "best_model_epoch_beforeFinetune.pth") 
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Expected fine-tuned model not found at {ckpt_path}")
    model = load_checkpoint(model, ckpt_path).to(device)

    print("Using Focal Loss for further training...")
    """— unfreeze ResNet —"""
    criterion = FocalLoss(gamma=2.0, ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)

    """— tracking lists —"""
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0

    print("Training with Focal Loss")
    for epoch in range(3):
        model.train()
        running_loss, running_correct, total_samples = 0.0, 0, 0
        batch_idx = 0
        
        """— training loop —"""
        for batch in train_loader:
            batch_idx += 1
            images, questions, masks, targets = [x.to(device) for x in batch]
            if torch.all(targets == -1): continue

            """ skip batch if all targets are -1 (ignore_index) after potential filtering"""
            optimizer.zero_grad()
            outputs = model(images, questions, masks)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            """ consider only valid targets for accuracy calculation (not -1) """
            valid_mask = (targets != -1)
            num_valid = valid_mask.sum().item()
            running_loss += loss.item() * num_valid
            running_correct += (outputs.argmax(1)[valid_mask] == targets[valid_mask]).sum().item()
            total_samples += num_valid
            
            """ print progress every 50 batches """
            if batch_idx % 50 == 0:
                batch_loss = loss.item()
                batch_acc = (outputs.argmax(1)[valid_mask] == targets[valid_mask]).sum().item() / num_valid
                print(f"[Focal Epoch {epoch+1}/3][Batch {batch_idx}/{len(train_loader)}] "
                      f"Batch Loss: {batch_loss:.4f}, Batch Acc: {batch_acc:.4f}")

        """ end of epoch: calculate average loss and accuracy """
        train_loss = running_loss / total_samples
        train_acc = running_correct / total_samples
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[Focal Epoch {epoch+1}/5] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        """ save if improved """
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, os.path.join(args.save_path, "focal_best.pth"))
            print(f"Saved improved model with Focal Loss at epoch {epoch+1}")

    """Plot curves"""
    plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir=args.save_path)

    """Final evaluation"""
    print("\nEvaluating best focal model on testA...")
    model = load_checkpoint(model, os.path.join(args.save_path, "focal_best.pth")).to(device)
    test_data = CLEVRVQADataset(args.dataset, split="testA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN,
                                 answer_to_idx=shared_answer_to_idx,
                                 idx_to_answer=shared_idx_to_answer,
                                 precomputed_num_answers=num_classes)
    """— create DataLoader for testA —"""
    test_loader = DataLoader(test_data, batch_size=64)
    evaluate_model(model, test_loader, criterion, device, full_report=True)
    visualize_preds(model, test_data, tokenizer, device, out_dir=args.save_path, correct=True, count=5, title="Correct_Focal")
    visualize_preds(model, test_data, tokenizer, device, out_dir=args.save_path, correct=False, count=5, title="Incorrect_Focal")

def zero_shot_evaluation(args):
    """Zero-shot evaluation on testB using the model trained on trainA and valA."""
    """— load shared vocab from trainA —"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for zero-shot evaluation")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    """Load vocab from trainA to ensure answer indices match"""
    print("Loading trainA to build answer vocabulary...")
    train_data = CLEVRVQADataset(args.dataset, split="trainA", tokenizer=tokenizer, max_q_len=MAX_Q_LEN)
    shared_answer_to_idx = train_data.answer_to_idx
    shared_idx_to_answer = train_data.idx_to_answer
    num_classes = train_data.num_answers

    """— load testB data with shared vocab —"""
    print("Initializing model and loading best trained weights...")
    model = VQAModel(vocab_size=len(tokenizer), num_classes=num_classes, max_len=MAX_Q_LEN)
    ckpt_path = os.path.join(args.pretrained_model_path, "best_model_epoch_beforeFinetune.pth") 
    
    """Load the checkpoint. This should be the best model from training."""
    model = load_checkpoint(model, ckpt_path).to(device)
    model.eval()

    print("Loading testB data...")
    """— load testB data with shared vocab —"""
    test_data = CLEVRVQADataset(args.dataset, split="testB", tokenizer=tokenizer, max_q_len=MAX_Q_LEN,
                                 answer_to_idx=shared_answer_to_idx,
                                 idx_to_answer=shared_idx_to_answer,
                                 precomputed_num_answers=num_classes)
    test_loader = DataLoader(test_data, batch_size=64, num_workers= 2)

    """— create DataLoader for testB —"""
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    """— evaluate on testB —"""
    print("\nEvaluating on Zero-Shot Test Set (testB):")
    evaluate_model(model, test_loader, criterion, device, full_report=True)

    print("\nVisualizing predictions on testB...")
    """— visualize predictions on testB —"""
    visualize_preds(model, test_data, tokenizer, device, out_dir=args.save_path, correct=True, count=5, title="Correct_TestB")
    visualize_preds(model, test_data, tokenizer, device, out_dir=args.save_path, correct=False, count=5, title="Incorrect_TestB")


if __name__ == '__main__':
    """Set up argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'finetune', 'focal', 'zeroshot'], required=True) #
    parser.add_argument('--dataset', type=str, required=True, help="Root directory of the CLEVR dataset") #
    parser.add_argument('--save_path', type=str, default='vqa_output', 
                        help="Directory to save model checkpoints, plots, and other outputs") #
    parser.add_argument('--model_path', type=str, help='Path to saved model checkpoint (for inference)') #
    parser.add_argument('--pretrained_model_path', type= str, help = 'Path to pretrained model(for Fine tuning)')
    args = parser.parse_args()



    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'finetune':
        finetune_image_encoder(args)
    elif args.mode == 'focal':
        train_with_focal_loss(args)
    elif args.mode == 'zeroshot':
        if not args.pretrained_model_path:
            parser.error("--model_path is required for zero-shot mode.")
        zero_shot_evaluation(args)
    elif args.mode == 'inference':
        if not args.model_path:
            parser.error("--model_path is required for inference mode.")
        inference(args)
        
        