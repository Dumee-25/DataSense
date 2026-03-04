import os


class DataValidator:
    def __init__(self, max_file_size_mb: int = 50):
        self.max_file_size_mb = max_file_size_mb

    def validate(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            return {'valid': False, 'error': 'File not found.'}

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            return {'valid': False, 'error': f'File too large ({size_mb:.1f} MB). Max is {self.max_file_size_mb} MB.'}

        if not file_path.lower().endswith('.csv'):
            return {'valid': False, 'error': 'Only CSV files are supported.'}

        return {'valid': True}
