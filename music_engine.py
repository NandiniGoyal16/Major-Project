import torch
import numpy as np
import os
import sys
import pickle
from pathlib import Path

# Add EMOPIA transformer to path
EMOPIA_PATH = Path(__file__).parent / "EMOPIA"
sys.path.insert(0, str(EMOPIA_PATH / "workspace" / "transformer"))

from models import TransformerModel
from utils import write_midi

class MusicEngine:
    def __init__(self, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.dict_path = EMOPIA_PATH / "dataset" / "co-representation" / "dictionary.pkl"
        self.dict_path = EMOPIA_PATH / "dataset" / "co-representation" / "dictionary.pkl"
        # User requested pretrained_transformer folder
        self.model_path = Path(__file__).parent / "pretrained_transformer" / "loss_25_params.pt"
        
        self.dictionary = self._load_dict()
        self.net = self._load_model()

    def _load_dict(self):
        with open(self.dict_path, "rb") as f:
            dictionary = pickle.load(f)
        return dictionary

    def _load_model(self):
        event2word, _ = self.dictionary
        n_class = [len(event2word[key]) for key in event2word.keys()]
        
        net = TransformerModel(n_class, is_training=False)
        
        # Ensure path exists before loading
        load_path = self.model_path
        if not load_path.exists():
             # Fallback to EMOPIA exp path if root one missing
             load_path = EMOPIA_PATH / "exp" / "pretrained_transformer" / "loss_25_params.pt"

        try:
            state_dict = torch.load(load_path, map_location='cpu')
        except Exception:
            state_dict = torch.load(load_path, map_location='cpu', weights_only=False)
        
        net.load_state_dict(state_dict)
        net.to(self.device)
        net.eval()
        print(f"Loaded Transformer from {load_path}")
        return net

    def _map_valence_arousal_to_tag(self, valence, arousal):
        if valence >= 0:
            return 1 if arousal >= 0 else 4
        else:
            return 2 if arousal >= 0 else 3

    def generate(self, valence, arousal, output_midi_path, emotion_label=None, duration_sec=15):
        emotion_tag = self._map_valence_arousal_to_tag(valence, arousal)
        
        # Final generation logic: Single call with forced timeline progression
        target_tokens = int(duration_sec * 500)  # max tokens for duration
        
        # If no label provided, infer a simple one for the fallback
        if emotion_label is None:
            label_map = {1: "Happy", 2: "Angry", 3: "Sad", 4: "Neutral"}
            emotion_label = label_map.get(emotion_tag, "Neutral")

        original_cuda = torch.Tensor.cuda
        if self.device == 'cpu':
            torch.Tensor.cuda = lambda self: self
        
        try:
            print(f"Generating music for {duration_sec}s (Target: {target_tokens} tokens)...")
            res, _ = self.net.inference_from_scratch(
                self.dictionary, emotion_tag, n_token=8, display=True,
                max_steps=target_tokens
            )
            
            if res is not None:
                res = res[:target_tokens] # Trim to target
                _, word2event = self.dictionary
                types = [word2event['type'][w[3]] for w in res]
                note_count = types.count('Note')
                print(f"Final generation complete: {len(res)} tokens. Note count: {note_count}")
                
                if note_count < 10: # Threshold for fallback
                    print(f"Insufficient notes ({note_count}). Applying enhanced fallback...")
                    fallback_res = []
                    if emotion_label in ["Happy", "Surprise"]:
                        pitches = [39, 43, 46, 50, 51, 55, 58, 62] # Major 9th
                    elif emotion_label in ["Sad"]:
                        pitches = [39, 42, 46, 49, 51, 54, 58, 61] # Minor 9th
                    elif emotion_label in ["Angry", "Fear", "Disgust"]:
                        pitches = [39, 42, 45, 48, 51, 54, 57, 60] # Diminished
                    else: # Neutral/Calm
                        pitches = [39, 41, 43, 46, 51, 53, 55, 58] # Suspended
                        
                    # Build long sequence for 30s
                    repeats = int(duration_sec * 4) + 2
                    for bar in range(repeats):
                        fallback_res.append([0, 0, 0, 2, 0, 0, 0, 0]) # Bar Start
                        for i, p in enumerate(pitches):
                            fallback_res.append([0, 0, (i*2)%16, 0, p, 10, 20, 0])
                
                write_midi(res, str(output_midi_path), word2event)
                return True
            return False
        finally:
            torch.Tensor.cuda = original_cuda
            
    def close(self):
        pass

if __name__ == "__main__":
    engine = MusicEngine()
    out_path = "test_output.mid"
    success = engine.generate(0.5, 0.5, out_path, "Happy")
    if success:
        print(f"Generated MIDI at {out_path}")
    else:
        print("Generation failed.")
