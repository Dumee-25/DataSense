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

        # Verify the file contains valid CSV-like content
        try:
            with open(file_path, 'rb') as f:
                head = f.read(8192)
            if not head:
                return {'valid': False, 'error': 'File is empty.'}
            # Check for null bytes (binary file)
            if b'\x00' in head:
                return {'valid': False, 'error': 'File appears to be binary, not a valid CSV.'}
            # Try decoding as text
            try:
                text = head.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = head.decode('latin-1')
                except UnicodeDecodeError:
                    return {'valid': False, 'error': 'File encoding not supported. Use UTF-8 or Latin-1.'}
            # Check for delimiter in first few lines
            lines = text.split('\n')[:5]
            if not any(',' in line for line in lines):
                return {'valid': False, 'error': 'File does not appear to contain comma-separated values.'}
        except Exception:
            return {'valid': False, 'error': 'Could not read file for validation.'}

        return {'valid': True}
