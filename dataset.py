import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class CLEVRVQADataset(Dataset):
    """
    CLEVR VQA Dataset
    Args:
        root_dir (str): Directory with all the images and questions.
        split (str): One of 'trainA', 'trainB', 'val', 'test'.
        tokenizer: Tokenizer for encoding questions.
        max_q_len (int): Maximum length of the question.
        answer_to_idx (dict): Precomputed mapping from answers to indices.
        idx_to_answer (dict): Precomputed mapping from indices to answers.
        precomputed_num_answers (int): Precomputed number of unique answers.
    """
    def __init__(self, root_dir, split="trainA", tokenizer=None, max_q_len=30,
                 answer_to_idx=None, idx_to_answer=None, precomputed_num_answers=None): # Added args for shared vocab
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len

        self.img_dir = os.path.join(root_dir, "images", split)
        self.q_path = os.path.join(root_dir, "questions", f"CLEVR_{split}_questions.json")

        with open(self.q_path, 'r') as f:
            data = json.load(f)
        self.entries = data['questions']

        """Use provided vocabulary if available, otherwise build it"""
        if answer_to_idx is not None and idx_to_answer is not None and precomputed_num_answers is not None:
            self.answer_to_idx = answer_to_idx
            self.idx_to_answer = idx_to_answer
            self._num_answers = precomputed_num_answers
        else:
            self.answer_to_idx = {}
            self.idx_to_answer = {}
            self._build_answer_vocab()
            self._num_answers = len(self.answer_to_idx)

        """Ensure the number of answers is consistent across splits"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _build_answer_vocab(self):
        """Builds the answer vocabulary from the dataset entries."""
        idx = 0
        for entry in self.entries:
            ans = entry['answer']
            if ans not in self.answer_to_idx:
                self.answer_to_idx[ans] = idx
                self.idx_to_answer[idx] = ans
                idx += 1

    def __len__(self):
        """Returns the number of entries in the dataset."""
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_filename = entry['image_filename']
        img_path = os.path.join(self.img_dir, image_filename)
        image = self.transform(Image.open(img_path).convert("RGB"))

        question = entry['question']
        encoded = self.tokenizer(question, padding='max_length', truncation=True,
                                 max_length=self.max_q_len, return_tensors='pt')
        question_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        answer = entry['answer']
        """Use the shared vocabulary to get the target index"""
        if answer in self.answer_to_idx:
            target = self.answer_to_idx[answer]
        else:
            """This case means an answer in val/test is not in the train vocab."""
            """For the assignment, this might indicate an issue or all answers are expected in trainA."""
            """Using a special index like -1 and `ignore_index` in CrossEntropyLoss is one way to handle this."""
            print(f"Warning: Answer '{answer}' from split '{self.split}' not found in vocabulary. Using target -1.")
            target = -1 # Make sure your loss function (e.g., CrossEntropyLoss) handles this via ignore_index=-1
        
        return image, question_ids, attention_mask, torch.tensor(target, dtype=torch.long)


    @property
    def num_answers(self):
        return self._num_answers

    def decode_answer(self, idx):
        return self.idx_to_answer.get(idx, "<UNK_ANSWER>")