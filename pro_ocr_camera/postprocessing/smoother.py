from collections import Counter, deque

class TemporalSmoother:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = {} # {track_id: deque([texts])}

    def update(self, track_id, new_text):
        if not new_text:
            return self.get_best(track_id)
            
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.window_size)
            
        self.history[track_id].append(new_text)
        return self.get_best(track_id)

    def get_best(self, track_id):
        if track_id not in self.history or not self.history[track_id]:
            return ""
            
        # Majority voting
        counts = Counter(self.history[track_id])
        # Return the most common text that isn't empty
        most_common = counts.most_common(1)[0][0]
        return most_common

    def clear(self, track_id):
        if track_id in self.history:
            del self.history[track_id]
            
    def cleanup_old_tracks(self, active_ids):
        """Removes history for tracks that are no longer visible."""
        current_ids = list(self.history.keys())
        for tid in current_ids:
            if tid not in active_ids:
                # We could keep them for a while, but for now clear
                self.clear(tid)
