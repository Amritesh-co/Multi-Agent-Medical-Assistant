import os
import json
import base64
import re
from mimetypes import guess_type

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class ClassificationDecision(BaseModel):
    """Output structure for the decision agent."""
    image_type: str = Field(description="The type of the medical image or 'NON-MEDICAL'")
    reasoning: str = Field(description="The reasoning behind the classification")
    confidence: float = Field(description="The confidence score of the classification")

class ImageClassifier:
    """Uses GPT-4o Vision to analyze images and determine their type."""
    
    def __init__(self, vision_model):
        self.vision_model = vision_model
        self.json_parser = JsonOutputParser(pydantic_object=ClassificationDecision)
        
    def local_image_to_data_url(self, image_path: str) -> str:
        """
        Get the url of a local image
        """
        mime_type, _ = guess_type(image_path)

        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{base64_encoded_data}"
    
    def classify_image(self, image_path: str) -> str:
        """Analyzes the image to classify it as a medical image and determine its type."""
        print(f"[ImageAnalyzer] Analyzing image: {image_path}")
        
        format_instructions = self.json_parser.get_format_instructions()

        vision_prompt = [
            {"role": "system", "content": "You are an expert in medical imaging. You must respond with ONLY valid JSON, no other text."},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    f"Determine if this is a medical image. If it is, classify it as 'BRAIN MRI SCAN', 'CHEST X-RAY', 'SKIN LESION', or 'OTHER'. If it's not a medical image, return 'NON-MEDICAL'.\n\n{format_instructions}"
                )},
                {"type": "image_url", "image_url": {"url": self.local_image_to_data_url(image_path)}}
            ]}
        ]
        
        # Invoke LLM to classify the image
        response = self.vision_model.invoke(vision_prompt)
        response_text = response.content.strip()
        
        # Robust JSON extraction
        extracted = response_text
        try:
            # First try direct parsing
            return self.json_parser.parse(response_text)
        except Exception:
            # If direct parsing fails, try to extract JSON block or curly braces
            
            # Try to find JSON in markdown blocks first
            markdown_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if markdown_match:
                extracted = markdown_match.group(1)
            else:
                # Fallback to finding anything between curly braces
                json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if json_match:
                    extracted = json_match.group(1)
            
            # Handle the specific case of double curly braces {{ ... }}
            if extracted.startswith('{{') and extracted.endswith('}}'):
                extracted = extracted[1:-1].strip()

            try:
                return self.json_parser.parse(extracted)
            except Exception as e:
                print(f"[ImageAnalyzer] Warning: Failed to parse response: {e}")
                print(f"[ImageAnalyzer] Original response: {response_text}")
                print(f"[ImageAnalyzer] Extracted text: {extracted}")
                return {"image_type": "unknown", "reasoning": f"Failed to parse classification: {str(e)}", "confidence": 0.0}
