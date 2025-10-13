import os
import json
from typing import Dict, List, Optional
import logging
import base64
from PIL import Image
import io

# OpenAI import for image transcription
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccuracyEvaluator:
    def __init__(self, 
                 api_key: str, 
                 provider: str = "openai",
                 model_name: str = "gpt-4o"):
        """
        Initialize the CER evaluator with GPT-4o for reference transcription
        
        Args:
            api_key: OpenAI API key for GPT-4o
            provider: LLM provider ("openai")
            model_name: Model to use for transcription ("gpt-4o")
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        
        if OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
            logger.info("✅ CER Evaluator initialized with GPT-4o for reference transcription")
        else:
            logger.warning("⚠️ OpenAI not available or no API key provided - will use prediction as reference")    
    def evaluate_prediction(self, 
                          image_path: str, 
                          prediction: str, 
                          evaluation_type: str = "cer",
                          reference_text: str = None) -> Dict:
        """
        Evaluate prediction using Character Error Rate (CER)
        Uses GPT-4o to transcribe the image as reference text if not provided
        
        Args:
            image_path: Path to the handwritten image
            prediction: Final prediction from TrOCR + Gemini
            evaluation_type: Type of evaluation (always CER)
            reference_text: Optional reference text (if not provided, will transcribe image)
            
        Returns:
            Dictionary containing CER evaluation results
        """
        logger.info(f"🔍 Starting CER evaluation for prediction: '{prediction[:50]}...'")
        
        # Get reference text - transcribe image if not provided
        if not reference_text and image_path and os.path.exists(image_path):
            logger.info("📝 Transcribing image with GPT-4o to get reference text...")
            reference_text = self._transcribe_image_with_gpt4o(image_path)
            if not reference_text:
                logger.warning("⚠️ Failed to transcribe image, using prediction as reference")
                reference_text = prediction
        elif not reference_text:
            logger.warning("⚠️ No reference text provided and no valid image path, using prediction as reference")
            reference_text = prediction
        
        # Calculate CER metrics
        cer_metrics = self._calculate_cer(prediction, reference_text)
        
        # Create evaluation result
        evaluation_result = {
            "prediction": prediction,
            "reference_text": reference_text,
            "cer": cer_metrics["cer"],
            "character_accuracy": cer_metrics["character_accuracy"],
            "total_characters": cer_metrics["total_characters"],
            "correct_characters": cer_metrics["correct_characters"],
            "error_characters": cer_metrics["error_characters"],
            "substitution_errors": cer_metrics["substitution_errors"],
            "insertion_errors": cer_metrics["insertion_errors"],
            "deletion_errors": cer_metrics["deletion_errors"],
            "edit_distance": cer_metrics["edit_distance"],
            "evaluation_type": "character_error_rate",
            "evaluation_success": True,
            "model_used": "gpt-4o_reference_cer_calculator",
            "reference_source": "gpt-4o_transcription" if reference_text != prediction else "prediction_fallback"
        }
        
        logger.info(f"✅ CER evaluation completed")
        logger.info(f"   Reference: '{reference_text[:50]}...'")
        logger.info(f"   CER: {cer_metrics['cer']:.4f} ({(cer_metrics['cer'] * 100):.2f}%)")
        logger.info(f"   Character Accuracy: {cer_metrics['character_accuracy']:.4f} ({(cer_metrics['character_accuracy'] * 100):.2f}%)")
        
        return evaluation_result
    
    def _transcribe_image_with_gpt4o(self, image_path: str) -> Optional[str]:
        """
        Transcribe image using GPT-4o Vision
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Transcribed text or None if failed
        """
        if not OPENAI_AVAILABLE or not self.api_key:
            logger.error("❌ OpenAI not available or no API key")
            return None
        
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare the request
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please transcribe the handwritten text in this image exactly as written. Return only the transcribed text without any additional commentary or formatting."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            transcribed_text = response.choices[0].message.content.strip()
            logger.info(f"📝 GPT-4o transcription: '{transcribed_text[:100]}...'")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"❌ Failed to transcribe image with GPT-4o: {e}")
            return None
    
    def _calculate_cer(self, prediction: str, reference_text: str = None) -> Dict:
        """
        Calculate Character Error Rate (CER) and related metrics
        
        Args:
            prediction: The predicted text
            reference_text: Reference text for comparison
            
        Returns:
            Dictionary containing CER metrics
        """
        if not reference_text:
            # If no reference text, return basic metrics
            return {
                "cer": 1.0,  # 100% error rate when no reference
                "character_accuracy": 0.0,
                "total_characters": len(prediction),
                "correct_characters": 0,
                "error_characters": len(prediction),
                "substitution_errors": 0,
                "insertion_errors": 0,
                "deletion_errors": 0,
                "edit_distance": len(prediction),
                "method": "no_reference"
            }
        
        # Normalize texts for comparison
        pred_norm = self._normalize_text(prediction)
        ref_norm = self._normalize_text(reference_text)
        
        # Calculate edit distance using Levenshtein algorithm
        edit_distance = self._calculate_edit_distance(pred_norm, ref_norm)
        
        # Calculate CER: edit_distance / max(len(prediction), len(reference))
        max_length = max(len(pred_norm), len(ref_norm))
        cer = edit_distance / max_length if max_length > 0 else 1.0
        
        # Character accuracy = 1 - CER
        character_accuracy = 1.0 - cer
        
        # Count correct characters
        correct_chars = max_length - edit_distance
        
        # Calculate error types (simplified)
        substitution_errors = min(edit_distance, min(len(pred_norm), len(ref_norm)))
        insertion_errors = max(0, len(pred_norm) - len(ref_norm))
        deletion_errors = max(0, len(ref_norm) - len(pred_norm))
        
        return {
            "cer": cer,
            "character_accuracy": character_accuracy,
            "total_characters": max_length,
            "correct_characters": correct_chars,
            "error_characters": edit_distance,
            "substitution_errors": substitution_errors,
            "insertion_errors": insertion_errors,
            "deletion_errors": deletion_errors,
            "edit_distance": edit_distance,
            "method": "levenshtein_distance"
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison (remove extra spaces, lowercase)"""
        if not text:
            return ""
        # Remove extra whitespace and convert to lowercase
        normalized = " ".join(text.lower().split())
        return normalized
    
    def _calculate_edit_distance(self, str1: str, str2: str) -> int:
        """Calculate Levenshtein edit distance between two strings"""
        if len(str1) < len(str2):
            return self._calculate_edit_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def batch_evaluate(self, 
                      predictions: List[str], 
                      image_paths: List[str], 
                      evaluation_type: str = "cer",
                      reference_texts: List[str] = None) -> List[Dict]:
        """
        Evaluate multiple predictions in batch using CER with GPT-4o reference transcription
        
        Args:
            predictions: List of prediction texts
            image_paths: List of corresponding image paths
            evaluation_type: Type of evaluation (always CER)
            reference_texts: Optional list of reference texts (if not provided, will transcribe images)
            
        Returns:
            List of CER evaluation results
        """
        results = []
        
        for i, prediction in enumerate(predictions):
            logger.info(f"Evaluating prediction {i+1}/{len(predictions)} using CER with GPT-4o reference")
            
            # Get reference text if available
            reference_text = None
            if reference_texts and i < len(reference_texts):
                reference_text = reference_texts[i]
            
            # Get image path
            image_path = ""
            if image_paths and i < len(image_paths):
                image_path = image_paths[i]
            
            result = self.evaluate_prediction(
                image_path=image_path,
                prediction=prediction, 
                evaluation_type=evaluation_type,
                reference_text=reference_text
            )
            results.append(result)
        
        return results
    
    def get_evaluation_summary(self, evaluations: List[Dict]) -> Dict:
        """Generate summary statistics from CER evaluations"""
        if not evaluations:
            return {}
        
        successful_evaluations = [e for e in evaluations if e.get("evaluation_success", False)]
        
        if not successful_evaluations:
            return {"error": "No successful evaluations found"}
        
        # CER metrics
        cer_values = [e.get("cer", 1.0) for e in successful_evaluations]
        accuracy_values = [e.get("character_accuracy", 0.0) for e in successful_evaluations]
        edit_distances = [e.get("edit_distance", 0) for e in successful_evaluations]
        
        summary = {
            "total_predictions": len(evaluations),
            "successful_evaluations": len(successful_evaluations),
            "average_cer": sum(cer_values) / len(cer_values),
            "average_character_accuracy": sum(accuracy_values) / len(accuracy_values),
            "average_edit_distance": sum(edit_distances) / len(edit_distances),
            "min_cer": min(cer_values),
            "max_cer": max(cer_values),
            "min_accuracy": min(accuracy_values),
            "max_accuracy": max(accuracy_values),
            "evaluation_success_rate": len(successful_evaluations) / len(evaluations) * 100
        }
        
        # CER distribution
        cer_distribution = {
            "excellent_0_5": len([c for c in cer_values if c <= 0.05]),  # 0-5% CER
            "very_good_5_10": len([c for c in cer_values if 0.05 < c <= 0.10]),  # 5-10% CER
            "good_10_20": len([c for c in cer_values if 0.10 < c <= 0.20]),  # 10-20% CER
            "fair_20_30": len([c for c in cer_values if 0.20 < c <= 0.30]),  # 20-30% CER
            "poor_above_30": len([c for c in cer_values if c > 0.30])  # >30% CER
        }
        summary["cer_distribution"] = cer_distribution
        
        return summary
    
    def test_model_capabilities(self) -> Dict:
        """Test the CER evaluator capabilities"""
        return {
            "model_name": "gpt-4o_reference_cer_calculator",
            "provider": "openai",
            "status": "operational" if OPENAI_AVAILABLE and self.api_key else "needs_api_key",
            "capabilities": {
                "character_error_rate": True,
                "edit_distance": True,
                "batch_evaluation": True,
                "multimodal": True,
                "vision_support": True,
                "gpt4o_transcription": True
            },
            "evaluation_type": "character_error_rate_with_gpt4o_reference"
        }
    
    def get_available_models(self) -> Dict:
        """Get information about available evaluation methods"""
        return {
            "current_model": "gpt-4o_reference_cer_calculator",
            "available_methods": {
                "gpt-4o_reference_cer": {
                    "provider": "openai",
                    "description": "Character Error Rate calculation using GPT-4o transcribed reference text",
                    "metrics": ["cer", "character_accuracy", "edit_distance"],
                    "reference_source": "gpt-4o_vision_transcription"
                }
            },
            "recommendation": "Use GPT-4o for high-quality reference transcription, then calculate CER"
        }

# Example usage
# if __name__ == "__main__":
#     # Test the CER evaluator with GPT-4o
#     evaluator = AccuracyEvaluator(api_key="your-openai-api-key")
    
#     # Test evaluation
#     test_result = evaluator.evaluate_prediction(
#         image_path="test_image.jpg",
#         prediction="Hello World",
#         reference_text=None  # Will transcribe image with GPT-4o
#     )
    
#     print("CER Evaluation Result:")
#     print(json.dumps(test_result, indent=2))
