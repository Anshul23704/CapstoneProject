import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor

try:
    from .tokenizer import CharacterTokenizer
except ImportError:
    from tokenizer import CharacterTokenizer


class TrOCRInferenceEngine:
    """
    High-performance inference engine for fine-tuned TrOCR ALPR models.
    Supports single-image and batched GPU inference directly on:
      - OpenCV BGR NumPy arrays (standard video frame crops)
      - PIL RGB Images
      - Image file paths (strings)
    """
    def __init__(self, model_path="model_checkpoint", device=None, use_fp16=True):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CharacterTokenizer()
        self.image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-small-printed")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TrOCR checkpoint directory not found at: {model_path}")
            
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        # Enable FP16 for ~2x GPU inference acceleration and 50% VRAM reduction
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        if self.use_fp16:
            self.model.half()
            
        self.model.to(self.device)
        self.model.eval()

    def _prepare_image(self, img_input):
        if isinstance(img_input, str):
            return Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            # OpenCV frames are default BGR; convert to RGB PIL Image
            if len(img_input.shape) == 3 and img_input.shape[2] == 3:
                return Image.fromarray(img_input[:, :, ::-1])
            return Image.fromarray(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img_input)}. Expected ndarray, PIL Image, or str path.")

    def predict(self, image_input):
        """
        Runs single-image OCR inference.

        Args:
            image_input: OpenCV BGR ndarray, PIL Image, or file path str.

        Returns:
            predicted_string (str): Clean license plate characters (e.g. 'MH12AB1234')
            avg_confidence (float): Mean confidence across predicted characters
            char_confidences (list of float): Softmax probability per character position
        """
        pil_image = self._prepare_image(image_input)
        pixel_values = self.image_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        if self.use_fp16:
            pixel_values = pixel_values.half()

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=12,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )

        token_sequence = outputs.sequences[0][1:]
        predicted_string = self.tokenizer.decode(token_sequence.tolist())

        char_confidences = []
        for step, logits in enumerate(outputs.scores):
            tok_id = token_sequence[step].item()
            if tok_id == self.tokenizer.eos_token_id:
                break
            if tok_id not in (self.tokenizer.pad_token_id, self.tokenizer.bos_token_id):
                probs = F.softmax(logits[0], dim=-1)
                char_confidences.append(round(probs[tok_id].item(), 4))

        avg_confidence = round(sum(char_confidences) / len(char_confidences), 4) if char_confidences else 0.0
        return predicted_string, avg_confidence, char_confidences

    def predict_batch(self, image_inputs):
        """
        Runs batched GPU inference across multiple plate crops in a single forward pass.
        Ideal for Stage 6/7 processing of vehicle tracks (up to 15 top-K crops).

        Args:
            image_inputs (list): List of OpenCV BGR ndarrays, PIL Images, or file paths.

        Returns:
            list of tuple: [(predicted_string, avg_confidence, char_confidences), ...]
        """
        if not image_inputs:
            return []

        pil_images = [self._prepare_image(img) for img in image_inputs]
        pixel_values = self.image_processor(images=pil_images, return_tensors="pt").pixel_values.to(self.device)
        if self.use_fp16:
            pixel_values = pixel_values.half()

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=12,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )

        batch_results = []
        batch_size = pixel_values.shape[0]

        for b in range(batch_size):
            token_sequence = outputs.sequences[b][1:]
            pred_text = self.tokenizer.decode(token_sequence.tolist())

            char_confs = []
            for step, logits in enumerate(outputs.scores):
                tok_id = token_sequence[step].item()
                if tok_id == self.tokenizer.eos_token_id:
                    break
                if tok_id not in (self.tokenizer.pad_token_id, self.tokenizer.bos_token_id):
                    probs = F.softmax(logits[b], dim=-1)
                    char_confs.append(round(probs[tok_id].item(), 4))

            avg_conf = round(sum(char_confs) / len(char_confs), 4) if char_confs else 0.0
            batch_results.append((pred_text, avg_conf, char_confs))

        return batch_results
