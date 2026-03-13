import os
import sys
from pathlib import Path
import miditoolkit
import numpy as np
import random
import librosa
import soundfile as sf
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore')

class AudioEngine:
    def __init__(self, dataset_root="music_dataset"):
        self.root = Path(dataset_root)
        self.genres = ["Western Music", "Indian Classical "]
        self.max_samples = 100 # User constraint
        self.sr = 22050 # Lower SR for speed
        
        # Load sample maps for each genre/instrument
        self.sample_maps = self._map_dataset()

    def _map_dataset(self):
        sample_maps = {}
        for genre in self.genres:
            genre_name = genre.strip()
            genre_path = self.root / genre
            if not genre_path.exists():
                print(f"Warning: Genre path {genre_path} not found.")
                continue
                
            sample_maps[genre_name] = {}
            for inst_dir in genre_path.iterdir():
                if inst_dir.is_dir():
                    # Get up to 100 wav/mp3 files
                    samples = sorted(list(inst_dir.glob("*.wav")) + list(inst_dir.glob("*.mp3")))
                    if samples:
                        # Strip trailing spaces from instrument directory names
                        clean_inst_name = inst_dir.name.strip()
                        # Use a deterministic sort so MIDI/Sample mapping remains stable
                        sample_maps[genre_name][clean_inst_name] = samples[:self.max_samples]
        return sample_maps

    def synthesize(self, midi_path, genre, instrument, output_wav_path, duration_sec=15):
        genre = genre.strip()
        if genre not in self.sample_maps or instrument not in self.sample_maps[genre]:
            raise ValueError(f"Instrument {instrument} in genre {genre} not found in dataset mapping.")
            
        samples_paths = self.sample_maps[genre][instrument]
        midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
        
        # Determine total duration in samples
        total_samples = int(duration_sec * self.sr)
        final_audio = np.zeros(total_samples)
        
        # Pre-load/cache samples to avoid repeated IO (limit to avoid memory explosion)
        # We'll load them on demand but keep them in a small cache if needed.
        loaded_samples = {}

        # Heuristic: MIDI Pitch mapping
        # We map indices 0-N of our samples to MIDI pitches 21-120 (standard piano range)
        # Note: This is an approximation since we don't know the actual pitch of the clips.
        
        print(f"Synthesizing {instrument} ({genre}) for {duration_sec}s...")
        
        ticks_per_beat = midi_obj.ticks_per_beat
        # Tempo is usually in the first tempo change or default 120
        tempo = 120
        if midi_obj.tempo_changes:
            tempo = midi_obj.tempo_changes[0].tempo

        # Convert ticks to seconds
        def tick_to_sec(tick):
            return (tick / ticks_per_beat) * (60.0 / tempo)

        for track in midi_obj.instruments:
            for note in track.notes:
                start_sec = tick_to_sec(note.start)
                if start_sec >= duration_sec:
                    continue
                
                # Map pitch to a sample index
                # Most datasets start around MIDI 21 or 36
                # We'll use a modulo/offset strategy
                sample_idx = (note.pitch - 21) % len(samples_paths)
                
                # Load sample
                s_path = samples_paths[sample_idx]
                if s_path not in loaded_samples:
                    # Load only first 2 seconds of each sample to keep it fast
                    audio, _ = librosa.load(str(s_path), sr=self.sr, duration=2.0)
                    loaded_samples[s_path] = audio
                
                clip = loaded_samples[s_path]
                
                # Placement
                start_idx = int(start_sec * self.sr)
                end_idx = min(start_idx + len(clip), total_samples)
                
                if start_idx < total_samples:
                    overlap_len = end_idx - start_idx
                    # Mix with existing audio (simple add)
                    # We scale by velocity (0-127)
                    scaled_clip = clip[:overlap_len] * (note.velocity / 127.0)
                    final_audio[start_idx:end_idx] += scaled_clip

        # Normalize final audio
        max_val = np.max(np.abs(final_audio))
        if max_val > 0:
            final_audio = final_audio / max_val * 0.9

        # Save to WAV
        sf.write(output_wav_path, final_audio, self.sr)
        return True

    def _create_dummy_wav(self, path):
        sr = 22050
        t = np.linspace(0, 1, sr)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        sf.write(path, audio, sr)

if __name__ == "__main__":
    # Test
    engine = AudioEngine()
    print("Available Instruments:", list(engine.sample_maps.keys()))
