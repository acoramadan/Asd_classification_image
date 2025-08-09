import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import cv2
from collections import Counter
import json

class EmotionDataset:
    def __init__(
        self,
        data_folder: str,
        processor: AutoProcessor,
        mode: str = "train", 
        emotion_labels: List[str] = None,
        max_length: int = 512
    ):
        self.data_folder = Path(data_folder)
        self.processor = processor
        self.mode = mode
        self.max_length = max_length
        
        self.emotion_labels = emotion_labels or [
            'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
        ]
        
        print(f"Initializing EmotionDataset in '{mode}' mode")
        print(f"Data folder: {data_folder}")
        
        if mode == "train":
            self.samples = self._load_training_data()
        elif mode == "test":
            self.samples = self._load_test_data()
        else:
            raise ValueError(f"Mode must be 'train' or 'test', got '{mode}'")
        
        print(f"Loaded {len(self.samples)} samples")
        self._show_statistics()

    def _load_training_data(self) -> List[Dict]:
       
        print("Loading training data...")
        samples = []        
        for emotion_folder in self.data_folder.iterdir():
            if not emotion_folder.is_dir():
                continue
                
            emotion = emotion_folder.name.lower()            
            if emotion not in self.emotion_labels:
                print(f"Skipping unknown emotion: {emotion}")
                continue
            
            print(f"Processing {emotion} folder...")            
            image_files = self._get_image_files(emotion_folder)
            
            for image_path in image_files:
                sample = {
                    'image_path': str(image_path),
                    'emotion': emotion,
                    'person_id': None
                }
                samples.append(sample)
            
            print(f"Found {len(image_files)} images")
        
        return samples
    
    def _load_test_data(self) -> List[Dict]:
        print("Loading test data...")
        samples = []
        
        for person_folder in self.data_folder.iterdir():
            if not person_folder.is_dir():
                continue
                
            person_id = person_folder.name
            print(f"Processing {person_id} folder...")
            
            image_files = self._get_image_files(person_folder)
            
            for image_path in image_files:
                sample = {
                    'image_path': str(image_path),
                    'emotion': None,
                    'person_id': person_id
                }
                samples.append(sample)   
            print(f"Found {len(image_files)} images")
        return samples
    
    def _get_image_files(self, folder: Path) -> List[Path]:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        return sorted(image_files)
    
    def _show_statistics(self):
        """Show dataset statistics"""
        
        if self.mode == "train":
            emotion_counts = Counter([s['emotion'] for s in self.samples])
            print(f"Training data statistics:")
            print(f"Total samples: {len(self.samples)}")
            print(f"Emotion distribution:")
            for emotion in self.emotion_labels:
                count = emotion_counts.get(emotion, 0)
                percentage = (count / len(self.samples)) * 100 if self.samples else 0
                print(f"     {emotion}: {count} samples ({percentage:.1f}%)")
        
        else:
            person_counts = Counter([s['person_id'] for s in self.samples])
            print(f"Test data statistics:")
            print(f"Total samples: {len(self.samples)}")
            print(f"Person distribution:")
            for person_id, count in person_counts.items():
                print(f"     {person_id}: {count} images")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.mode == "train":
            return self._get_training_item(sample)
        else:
            return self._get_test_item(sample)
        
    def _get_training_item(self, sample: Dict):
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {sample['image_path']}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        emotion = sample['emotion']        
        instruction = f"Classify the emotion in this facial expression image. Choose from: {', '.join(self.emotion_labels)}."
        expected_response = f"The emotion in this image is {emotion}."        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            },
            {
                "role": "assistant",
                "content": expected_response
            }
        ]
        
        text = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=False
        )
        
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': inputs['input_ids'].squeeze(0).clone(),
            'emotion': emotion,
            'image_path': sample['image_path']
        }
    
    def _get_test_item(self, sample: Dict):
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {sample['image_path']}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        return {
            'image': image,
            'image_path': sample['image_path'],
            'person_id': sample['person_id']
        }
    
    def get_person_samples(self, person_id: str) -> List[Dict]:
        """Get all samples untuk specific person (test mode only)"""
        
        if self.mode != "test":
            raise ValueError("get_person_samples only available in test mode")
        
        person_samples = [s for s in self.samples if s['person_id'] == person_id]
        return person_samples
    