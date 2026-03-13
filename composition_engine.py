import os
import random
import json
from pathlib import Path
from pydub import AudioSegment

# Explicitly set ffmpeg/ffprobe paths for Homebrew on M1/M2 Mac
if os.path.exists("/opt/homebrew/bin/ffmpeg"):
    AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"
    AudioSegment.ffprobe = "/opt/homebrew/bin/ffprobe"

class CompositionEngine:
    def __init__(self, dataset_path="music_dataset"):
        self.root = Path(dataset_path)
        self.history_file = Path("composition_history.json")
        self.history = self._load_history()
        # Discover and store both nested and flat maps
        self.genre_map = self._discover_instruments()
        # self.instruments is set inside _discover_instruments

    def _load_history(self):
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f)

    def _discover_instruments(self):
        # Result: { "Genre": { "Instrument": [files...] } }
        genre_map = {}
        print(f"Scanning for instruments in {self.root}...")
        for genre_path in self.root.iterdir():
            if genre_path.is_dir():
                genre_name = genre_path.name.strip()
                genre_map[genre_name] = {}
                print(f" Checking genre: {genre_name}")
                for inst_dir in genre_path.iterdir():
                    if inst_dir.is_dir():
                        files = list(inst_dir.glob("*.wav")) + list(inst_dir.glob("*.mp3"))
                        if files:
                            clean_name = inst_dir.name.strip()
                            genre_map[genre_name][clean_name] = [str(f) for f in files]
                            print(f"  Found instrument: {clean_name} ({len(files)} files)")
        
        # Also keep a flat map for backward compatibility or easy lookup
        self.instruments = {}
        for g in genre_map:
            self.instruments.update(genre_map[g])
            
        return genre_map

    def get_instruments_by_genre(self, genre):
        genre = genre.strip()
        return self.genre_map.get(genre, {})

    def compose(self, emotion, instruments, duration_sec=30):
        """
        Composes a target duration audio. 
        If multiple instruments are provided, they play SIMULTANEOUSLY.
        """
        if isinstance(instruments, str):
            instruments = [instruments]
        
        target_frame_rate = 44100
        target_ms = duration_sec * 1000
        
        # Start with silence
        master = AudioSegment.silent(duration=target_ms, frame_rate=target_frame_rate)
        
        # We'll build each instrument's timeline separately and then overlay them
        for instrument in instruments:
            instrument = instrument.strip()
            # Find which genre/path this instrument belongs to
            # (In this engine, self.instruments is the flat map of files)
            clips_pool = self.instruments.get(instrument)
            if not clips_pool:
                # Fuzzy match
                for k in self.instruments.keys():
                    if instrument.lower() == k.lower():
                        clips_pool = self.instruments[k]
                        instrument = k
                        break
            
            if not clips_pool:
                print(f"Skipping unknown instrument: {instrument}")
                continue

            # Create a full-duration track for THIS instrument
            inst_track = AudioSegment.empty()
            inst_files = clips_pool.copy()
            random.shuffle(inst_files)
            
            # Load history for THIS instrument to ensure variety
            history = self._load_history()
            used_files = history.get(instrument, [])
            unused = [f for f in inst_files if f not in used_files]
            if not unused:
                history[instrument] = []
                unused = inst_files
            
            work_pool = unused
            random.shuffle(work_pool)
            
            current_inst_ms = 0
            selected_for_inst = []
            
            # Concatenate clips until we fill the duration for THIS instrument
            while current_inst_ms < target_ms:
                if not work_pool:
                    work_pool = inst_files.copy()
                    random.shuffle(work_pool)
                
                f_path = work_pool.pop(0)
                try:
                    clip = AudioSegment.from_file(f_path).set_frame_rate(target_frame_rate)
                    clip = clip.normalize() # Ensure balanced volume
                    
                    if len(inst_track) > 0:
                        fade_ms = min(500, len(clip) // 4, len(inst_track) // 4)
                        try:
                            inst_track = inst_track.append(clip, crossfade=fade_ms)
                        except:
                            inst_track = inst_track + clip
                    else:
                        inst_track = clip
                    
                    selected_for_inst.append(f_path)
                    current_inst_ms = len(inst_track)
                except: continue
                
            # Trim and fade instrument track
            inst_track = inst_track[:target_ms].fade_out(1000)
            
            # Update history for this instrument
            for f in selected_for_inst:
                if f not in self.history.get(instrument, []):
                    if instrument not in self.history: self.history[instrument] = []
                    self.history[instrument].append(f)
            
            # OVERLAY onto master (this makes them play TOGETHER)
            master = master.overlay(inst_track)

        # Final normalization to ensure no clipping after overlays
        master = master.normalize()
        
        self._save_history()
        return master

if __name__ == "__main__":
    engine = CompositionEngine()
    print("Found instruments:", list(engine.instruments.keys()))
    # Test
    # audio = engine.compose("Happy", "Piano", 15)
    # audio.export("test_comp.wav", format="wav")
