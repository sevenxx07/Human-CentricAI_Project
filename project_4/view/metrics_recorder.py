import os
import json
import time
import uuid
from datetime import datetime
from django.conf import settings
import numpy as np
import pandas as pd


class MetricsRecorder:
    """Simple metrics recorder that saves data to text files"""

    def __init__(self, session_id=None, study_mode=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.study_mode = study_mode or 'unknown'
        self.metrics_dir = os.path.join(settings.BASE_DIR, 'metrics_data')
        self.filename = f"user_{self.session_id}_{self.study_mode}.txt"
        self.filepath = os.path.join(self.metrics_dir, self.filename)

        # Create metrics directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Session tracking
        self.session_start_time = time.time()
        self.ratings = []  # List of rating data
        self.skips = []  # List of skip data
        self.session_phase = "initial"  # "initial" or "active_learning"

        # Initialize the file with session info
        self._initialize_file()

    def _initialize_file(self):
        """Initialize the metrics file with session header"""
        with open(self.filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MOVIE RECOMMENDER METRICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Study Mode: {self.study_mode}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Start Timestamp: {self.session_start_time}\n")
            f.write("\n")

    def record_rating(self, movie_id, rating, movie_title=None, time_taken=None, predicted_rating=None):
        """Record a single rating with timing and prediction"""
        timestamp = time.time()
        elapsed_since_start = timestamp - self.session_start_time

        rating_data = {
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': timestamp,
            'elapsed_since_start': elapsed_since_start,
            'time_taken': time_taken,
            'movie_title': movie_title,
            'session_phase': self.session_phase,
            'predicted_rating': predicted_rating,  # This is the expected value!
            'has_prediction': predicted_rating is not None
        }

        self.ratings.append(rating_data)

        # Append to file immediately with better formatting
        with open(self.filepath, 'a') as f:
            f.write(f"RATING: {movie_id} -> {rating} | ")
            f.write(f"Title: {movie_title or 'Unknown'} | ")
            if predicted_rating is not None:
                error = abs(rating - predicted_rating)
                f.write(f"Expected: {predicted_rating:.3f} | Error: {error:.3f} | ")
            else:
                f.write("Expected: N/A | Error: N/A | ")
            f.write(f"Phase: {self.session_phase} | ")
            f.write(f"Time: {time_taken:.2f}s | " if time_taken else "Time: N/A | ")
            f.write(f"Elapsed: {elapsed_since_start:.1f}s\n")

    def record_skip(self, movie_id, movie_title=None, time_taken=None, predicted_rating=None):
        """Record a movie skip"""
        timestamp = time.time()
        elapsed_since_start = timestamp - self.session_start_time

        skip_data = {
            'movie_id': movie_id,
            'timestamp': timestamp,
            'elapsed_since_start': elapsed_since_start,
            'time_taken': time_taken,
            'movie_title': movie_title,
            'session_phase': self.session_phase,
            'predicted_rating': predicted_rating
        }

        self.skips.append(skip_data)

        # Append to file immediately
        with open(self.filepath, 'a') as f:
            f.write(f"SKIP: {movie_id} | ")
            f.write(f"Title: {movie_title or 'Unknown'} | ")
            f.write(f"Expected: {predicted_rating:.2f} | " if predicted_rating else "Expected: N/A | ")
            f.write(f"Phase: {self.session_phase} | ")
            f.write(f"Time: {time_taken:.2f}s | " if time_taken else "Time: N/A | ")
            f.write(f"Elapsed: {elapsed_since_start:.1f}s\n")

    def set_phase(self, phase):
        """Update current session phase"""
        self.session_phase = phase
        with open(self.filepath, 'a') as f:
            f.write(f"\n--- PHASE CHANGE: {phase.upper()} ---\n")

    def calculate_rmse_from_logged_data(self):
        """Calculate RMSE from already logged actual vs expected ratings"""
        # Only use ratings that have both actual rating and predicted rating
        valid_ratings = [r for r in self.ratings if r.get('predicted_rating') is not None]

        if len(valid_ratings) == 0:
            return None

        errors = []
        for rating_data in valid_ratings:
            actual = rating_data['rating']
            predicted = rating_data['predicted_rating']
            error = (actual - predicted) ** 2
            errors.append(error)

        rmse = np.sqrt(np.mean(errors)) if errors else None

        # Log some debug info
        if rmse is not None:
            print(f"RMSE calculated from {len(valid_ratings)} ratings: {rmse:.4f}")

        return rmse

    def finalize_session(self, V_matrix=None, R_matrix=None, user_vector=None):
        """Calculate final metrics and write summary"""
        session_end_time = time.time()
        total_session_time = session_end_time - self.session_start_time

        # Calculate metrics
        num_ratings = len(self.ratings)
        num_skips = len(self.skips)

        # Average time per rating (only for ratings with time_taken data)
        ratings_with_time = [r for r in self.ratings if r.get('time_taken') is not None]
        avg_time_per_rating = (
            np.mean([r['time_taken'] for r in ratings_with_time])
            if ratings_with_time else None
        )

        # Calculate RMSE from logged data (actual vs expected)
        rmse = self.calculate_rmse_from_logged_data()

        # Count how many ratings had predictions
        ratings_with_predictions = len([r for r in self.ratings if r.get('predicted_rating') is not None])

        # Write final summary
        with open(self.filepath, 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("FINAL METRICS SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Session Duration: {total_session_time:.1f} seconds\n")
            f.write(f"Number of Ratings: {num_ratings}\n")
            f.write(f"Number of Skips: {num_skips}\n")
            f.write(f"Ratings with Predictions: {ratings_with_predictions}/{num_ratings}\n")
            f.write(
                f"Average Time per Rating: {avg_time_per_rating:.2f}s\n" if avg_time_per_rating else "Average Time per Rating: N/A\n")
            f.write(f"RMSE: {rmse:.4f}\n" if rmse else "RMSE: N/A (no predictions available)\n")
            f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")

            # Write detailed breakdown
            initial_ratings = [r for r in self.ratings if r['session_phase'] == 'initial']
            active_ratings = [r for r in self.ratings if r['session_phase'] == 'active_learning']

            f.write("PHASE BREAKDOWN:\n")
            f.write(f"Initial Phase Ratings: {len(initial_ratings)}\n")
            f.write(f"Active Learning Ratings: {len(active_ratings)}\n")

            # Calculate RMSE per phase if possible
            if rmse is not None:
                initial_with_pred = [r for r in initial_ratings if r.get('predicted_rating') is not None]
                active_with_pred = [r for r in active_ratings if r.get('predicted_rating') is not None]

                if initial_with_pred:
                    initial_rmse = np.sqrt(
                        np.mean([(r['rating'] - r['predicted_rating']) ** 2 for r in initial_with_pred]))
                    f.write(f"Initial Phase RMSE: {initial_rmse:.4f} ({len(initial_with_pred)} ratings)\n")

                if active_with_pred:
                    active_rmse = np.sqrt(
                        np.mean([(r['rating'] - r['predicted_rating']) ** 2 for r in active_with_pred]))
                    f.write(f"Active Learning RMSE: {active_rmse:.4f} ({len(active_with_pred)} ratings)\n")

            f.write("\n")

            # Write individual rating details for analysis
            f.write("DETAILED RATING LOG:\n")
            for i, rating_data in enumerate(self.ratings, 1):
                f.write(f"{i}. Movie {rating_data['movie_id']}: ")
                f.write(f"Actual={rating_data['rating']}, ")
                if rating_data.get('predicted_rating') is not None:
                    pred = rating_data['predicted_rating']
                    error = abs(rating_data['rating'] - pred)
                    f.write(f"Expected={pred:.3f}, Error={error:.3f}")
                else:
                    f.write("Expected=N/A, Error=N/A")
                f.write(f" | Phase={rating_data['session_phase']}\n")

        return {
            'session_duration': total_session_time,
            'num_ratings': num_ratings,
            'num_skips': num_skips,
            'avg_time_per_rating': avg_time_per_rating,
            'rmse': rmse,
            'ratings_with_predictions': ratings_with_predictions
        }

    def get_current_metrics(self):
        """Get current session metrics without finalizing"""
        current_time = time.time()
        session_duration = current_time - self.session_start_time

        ratings_with_time = [r for r in self.ratings if r.get('time_taken') is not None]
        avg_time_per_rating = (
            np.mean([r['time_taken'] for r in ratings_with_time])
            if ratings_with_time else None
        )

        # Calculate current RMSE
        current_rmse = self.calculate_rmse_from_logged_data()

        return {
            'session_duration': session_duration,
            'num_ratings': len(self.ratings),
            'num_skips': len(self.skips),
            'avg_time_per_rating': avg_time_per_rating,
            'current_rmse': current_rmse
        }


class TimingTracker:
    """Helper class to track timing for individual actions"""

    def __init__(self):
        self.start_time = None
        self.action_start_times = {}

    def start_action(self, action_id):
        """Start timing for a specific action"""
        self.action_start_times[action_id] = time.time()

    def end_action(self, action_id):
        """End timing for a specific action and return duration"""
        if action_id in self.action_start_times:
            duration = time.time() - self.action_start_times[action_id]
            del self.action_start_times[action_id]
            return duration
        return None


def create_session_metrics_recorder(session_id, study_mode):
    """Factory function to create a metrics recorder for a session"""
    return MetricsRecorder(session_id, study_mode)