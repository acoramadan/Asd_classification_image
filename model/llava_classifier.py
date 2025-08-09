from typing import List, Dict, Optional, Tuple
import torch
import json
from .video_emotion_dataset import EmotionDataset
from pathlib import Path
from PIL import Image
from collections import Counter
from transformers import AutoProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments, BitsAndBytesConfig


class LLaVaEmotionClassifier:
    def __init__(
            self, 
            model_name:str = "llava-hf/llava-v1.6-mistral-7b-hf",
            emotion_labels: List[str] = None, 
            use_quantization: bool = True,
            optimize_for_6gb: bool = True
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.optimize_for_6gb = optimize_for_6gb
        self.emotion_labels = emotion_labels or [
            'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
        ]
        print(f"Initializing EmotionClassifier")
        print(f"Model: {model_name}")
        print(f"Emotions: {self.emotion_labels}")

        self._optimize_memory_settings()
        self._load_model() 
        print("EmotionClassifier ready!")

    
    def _optimize_memory_settings(self):
        torch.backends.cuda.enable_flash_sdp(True)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)            
            torch.cuda.memory._set_allocator_settings('expandable_segments:True')
    
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        if self.use_quantization and torch.cuda.is_available():

            if self.optimize_for_6gb:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                print("Using enhanced 8-bit quantization for 6GB VRAM")
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                print("Using standard 8-bit quantization")
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory={0: "5.2GB"}, 
                attn_implementation="flash_attention_2" if self._is_flash_attn_available() else "sdpa"
            )
        else:
            print("Quantization disabled - NOT recommended for 6GB VRAM")
            print("Model may exceed 6GB memory limit without quantization")
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                max_memory={0: "5.5GB"} 
            )
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("   ✅ Gradient checkpointing enabled")
        
        self.model.eval()        
        self._check_memory_usage()
    
    def _is_flash_attn_available(self) -> bool:
        """Check if flash attention is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _check_memory_usage(self):        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            usage_percent = (reserved / total) * 100
            
            print(f"      GPU Memory Status:")
            print(f"      Allocated: {allocated:.2f} GB")
            print(f"      Reserved: {reserved:.2f} GB")
            print(f"      Total: {total:.2f} GB")
            print(f"      Usage: {usage_percent:.1f}%")
            
            if usage_percent > 95:
                print(f"   Critical: Memory usage > 95% - Likely to OOM")
                print(f"   Consider using 4-bit quantization or smaller model")
            elif usage_percent > 85:
                print(f"   Warning: Memory usage > 85% - Monitor for OOM")
                print(f"   Reduce batch size during training/inference")
            elif usage_percent > 70:
                print(f"   Caution: Memory usage > 70% - Good but watch batch size")
            else:
                print(f"   Good: Memory usage is acceptable for 6GB GPU")

    def get_optimal_batch_sizes(self) -> dict:
        
        current_usage = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        
        if current_usage > 5.0:  
            return {
                "training_batch_size": 1,
                "inference_batch_size": 4,
                "fine_tuning_batch_size": 2
            }
        elif current_usage > 4.0: 
            return {
                "training_batch_size": 2,
                "inference_batch_size": 6,
                "fine_tuning_batch_size": 3
            }
        else: 
            return {
                "training_batch_size": 4,
                "inference_batch_size": 8,
                "fine_tuning_batch_size": 4
            }
    
    def clear_memory_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU memory cache cleared")
    
    def monitor_memory_during_inference(self, enable: bool = True):
        self.memory_monitoring = enable
        if enable:
            print("Memory monitoring enabled - will show usage during inference")

    def clear_memory_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU memory cache cleared")
    
    def check_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"GPU Memory: {reserved:.2f}GB used / {total:.2f}GB total ({reserved/total*100:.1f}%)")
            
            if reserved/total > 0.85:
                print("High memory usage - consider clearing cache or reducing batch size")

    def fine_tune(
        self,
        train_folder: str,
        val_folder: str = None,
        output_dir: str = "./fine_tuned_model",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5
    ): 
        print(f"Starting fine-tuning...")
        print(f"Train folder: {train_folder}")
        print(f"Val folder: {val_folder}")
        
        train_dataset = EmotionDataset(
            data_folder=train_folder,
            processor=self.processor,
            mode="train",
            emotion_labels=self.emotion_labels
        )
        
        val_dataset = None
        if val_folder:
            val_dataset = EmotionDataset(
                data_folder=val_folder,
                processor=self.processor,
                mode="train",
                emotion_labels=self.emotion_labels
            )
        def data_collator(features):
            batch = {}
            keys = ['input_ids', 'attention_mask', 'pixel_values', 'labels']
            
            for key in keys:
                batch[key] = torch.stack([f[key] for f in features])

            return batch
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=50,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True if val_dataset else False,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
        )
        print("Starting training...")
        trainer.train()
        
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)
        
        print(f"Fine-tuning completed! Model saved to {output_dir}")
    
    def predict_single_image(self, image: Image.Image) -> Dict:
        
        instruction = f"Classify the emotion in this facial expression image. Choose from: {', '.join(self.emotion_labels)}."
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        response = self.processor.batch_decode(
            generate_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        
        predicted_emotion = self._extract_emotion(response)
        
        return {
            'predicted_emotion': predicted_emotion,
            'raw_response': response
        }
    
    def _extract_emotion(self, response: str) -> str:
        """Extract emotion dari model response"""
        response_lower = response.lower()
        
        for emotion in self.emotion_labels:
            if emotion.lower() in response_lower:
                return emotion
        
        return 'neutral'
    
    def analyze_person_emotions(
        self, 
        test_folder: str, 
        person_id: str,
        batch_size: int = 8
    ) -> Dict:
        print(f"👤 Analyzing emotions for {person_id}...")        
        test_dataset = EmotionDataset(
            data_folder=test_folder,
            processor=self.processor,
            mode="test",
            emotion_labels=self.emotion_labels
        )
        person_samples = test_dataset.get_person_samples(person_id)
        if not person_samples:
            raise ValueError(f"No samples found for person: {person_id}")
        
        print(f"Found {len(person_samples)} images for {person_id}")        
        all_predictions = []
        
        for i in range(0, len(person_samples), batch_size):
            batch_samples = person_samples[i:i+batch_size]
            
            print(f"  Processing batch {i//batch_size + 1}/{(len(person_samples)-1)//batch_size + 1}")
            
            for sample in batch_samples:
                image = Image.open(sample['image_path']).convert('RGB')                
                result = self.predict_single_image(image)
                all_predictions.append(result['predicted_emotion'])
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        emotion_counts = Counter(all_predictions)
        total_images = len(all_predictions)        
        emotion_distribution = {}
        for emotion in self.emotion_labels:
            count = emotion_counts.get(emotion, 0)
            percentage = count / total_images
            emotion_distribution[emotion] = round(percentage, 3)
        
        result = {
            'person_id': person_id,
            'total_images': total_images,
            'emotion_counts': dict(emotion_counts),
            'emotion_distribution': emotion_distribution,
            'dominant_emotion': max(emotion_distribution.keys(), key=lambda x: emotion_distribution[x])
        }
        
        print(f"Results for {person_id}:")
        print(f"   {json.dumps(emotion_distribution, indent=2)}")
        
        return result
    
    def analyze_all_persons(
        self, 
        test_folder: str,
        output_dir: str = "./results"
    ) -> Dict[str, Dict]:
    
        print(f"Analyzing all persons in {test_folder}")

        test_path = Path(test_folder)
        person_folders = [d for d in test_path.iterdir() if d.is_dir()]
        
        if not person_folders:
            raise ValueError(f"No person folders found in {test_folder}")
        
        print(f"Found {len(person_folders)} persons to analyze")
        
        all_results = {}
        for person_folder in person_folders:
            person_id = person_folder.name
            
            try:
                result = self.analyze_person_emotions(
                    test_folder=test_folder,
                    person_id=person_id
                )
                all_results[person_id] = result
                
            except Exception as e:
                print(f"Error processing {person_id}: {e}")
                all_results[person_id] = {"error": str(e)}
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / "emotion_analysis_results.json", 'w') as f:
                json.dump(all_results, f, indent=2)
            
            self._create_summary_report(all_results, output_path)
            
            print(f"Results saved to {output_dir}")
        
        return all_results
    
    def _create_summary_report(self, results: Dict, output_path: Path):     
        report_path = output_path / "summary_report.txt"
        with open(report_path, 'w') as f:
            f.write("EMOTION ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            for person_id, result in results.items():
                f.write(f"Person: {person_id}\n")
                
                if "error" in result:
                    f.write(f"  ERROR: {result['error']}\n\n")
                    continue
                
                f.write(f"  Total Images: {result['total_images']}\n")
                f.write(f"  Dominant Emotion: {result['dominant_emotion']}\n")
                f.write(f"  Emotion Distribution:\n")
                
                for emotion, percentage in result['emotion_distribution'].items():
                    f.write(f"    {emotion}: {percentage:.3f} ({percentage*100:.1f}%)\n")
                
                f.write("\n")
        
        print(f"Summary report saved to {report_path}")
    