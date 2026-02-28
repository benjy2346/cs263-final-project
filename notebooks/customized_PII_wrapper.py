from typing import List, Optional, Dict
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from presidio_evaluator import InputSample, span_to_tag
from presidio_evaluator.models import BaseModel

class CustomizedDetector(BaseModel):
    def __init__(
        self,
        endpoint: str,
        key: str,
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        labeling_scheme: str = "IO",
        entity_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
            entity_mapping=entity_mapping,
        )
        self.name = "Azure PII Detector"
        # Initialize Azure client
        text_analytics_credential = AzureKeyCredential(key)
        self.client = TextAnalyticsClient(endpoint=endpoint, credential=text_analytics_credential)

    def predict(self, sample: InputSample, **kwargs) -> List[str]:
        # 1. Call Azure API
        try:
            # Note: Azure API supports batch operations, but presidio-evaluator defaults to single predict
            # The language parameter can be hardcoded here or retrieved from the sample as needed
            response = self.client.recognize_pii_entities(documents=[sample.full_text], language="en")[0]
        except Exception as e:
            if self.verbose:
                print(f"Error calling Azure API: {e}")
            # If error occurs, return all 'O' (Outside) labels
            return ["O"] * len(sample.tokens)

        # 2. Parse Azure response
        starts = []
        ends = []
        scores = []
        tags = []
        
        if not response.is_error:
            for entity in response.entities:
                starts.append(entity.offset)
                ends.append(entity.offset + entity.length)  # Azure returns length, convert to end
                
                # Azure labels (e.g., 'Person', 'PhoneNumber'), convert to uppercase for mapping
                azure_label = entity.category.upper() 
                
                # Entity mapping
                mapped_tag = self.entity_mapping.get(azure_label, azure_label) if self.entity_mapping else azure_label
                tags.append(mapped_tag)
                
                scores.append(entity.confidence_score)
                
        # 3. Convert to Evaluator token-level labels
        response_tags = span_to_tag(
            scheme=self.labeling_scheme,
            text=sample.full_text,
            starts=starts,
            ends=ends,
            tokens=sample.tokens,
            scores=scores,
            tags=tags,
        )
        return response_tags

# --- Initialize your Azure model ---
AZURE_ENDPOINT = "https://<your-resource-name>.cognitiveservices.azure.com/"
AZURE_KEY = "<your-api-key>"

# Instantiate
azure_model = CustomizedDetector(endpoint=AZURE_ENDPOINT, key=AZURE_KEY)