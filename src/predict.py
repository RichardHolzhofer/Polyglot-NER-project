import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class NERPredictor:
    def __init__(self, model_path, device=None):
        """
        Initialize the predictor with a trained model.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model and tokenizer from {model_path} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
        self.id2label = self.model.config.id2label

    def predict(self, text):
        """
        Predict entities in a raw string.
        Returns a list of dictionaries with 'word' and 'label'.
        """
        # Tokenize (handling split into words is simpler for alignment)
        words = text.split()
        inputs = self.tokenizer(
            words, 
            is_split_into_words=True, 
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
        
        # Get word IDs to align back to original words
        word_ids = inputs.word_ids()
        
        results = []
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            # Skip special tokens and follow-up sub-tokens
            if word_idx is None or word_idx == previous_word_idx:
                continue
            
            label_id = predictions[i]
            label = self.id2label[label_id]
            results.append({
                "word": words[word_idx],
                "label": label
            })
            previous_word_idx = word_idx
            
        return results

if __name__ == "__main__":
    # Example usage:
    # predictor = NERPredictor("models/polyglot-ner-xlmr")
    # results = predictor.predict("Wolf László az OTP Bank vezérigazgató-helyettese.")
    # for r in results:
    #     print(f"{r['word']}: {r['label']}")
    print("NERPredictor script ready. Use it by importing the class.")
