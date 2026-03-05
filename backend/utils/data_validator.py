import os
import csv
import logging

logger = logging.getLogger(__name__)

# Byte-sequences that signal a non-text / binary file (not a real CSV)
_BINARY_SIGNATURES = [
    b'\x89PNG',       # PNG
    b'\xff\xd8\xff',  # JPEG
    b'PK\x03\x04',   # ZIP / XLSX / DOCX
    b'%PDF',          # PDF
    b'\x00\x00',      # generic null bytes
    b'\x7fELF',       # ELF binary
    b'\xd0\xcf\x11',  # MS Office (OLE)
]

# Maximum number of rows we'll scan during the sniff phase
_SNIFF_ROWS = 50


class DataValidator:
    """
    Validates uploaded files before pipeline processing.

    Checks performed:
      1. File exists
      2. File size within limit
      3. Extension is .csv
      4. File is not binary / non-text
      5. File is parseable as CSV with at least 1 data row and 1 column
      6. File is not excessively wide (prevents memory bombs)
      7. Sniff for formula-injection patterns (=, +, -, @, |) in cell values
    """

    def __init__(
        self,
        max_file_size_mb: int = 50,
        max_columns: int = 2000,
        warn_formula_injection: bool = True,
    ):
        self.max_file_size_mb = max_file_size_mb
        self.max_columns = max_columns
        self.warn_formula_injection = warn_formula_injection

    # ── public API ────────────────────────────────────────────────────────

    def validate(self, file_path: str) -> dict:
        """
        Run all validation checks. Returns::

            { "valid": True/False,
              "error": "..." | None,
              "warnings": [ ... ] }
        """
        warnings: list[str] = []

        # 1) Existence
        if not os.path.exists(file_path):
            return self._fail("File not found.")

        # 2) Size
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            return self._fail(
                f"File too large ({size_mb:.1f} MB). Max is {self.max_file_size_mb} MB."
            )

        # 3) Extension
        if not file_path.lower().endswith(".csv"):
            return self._fail("Only CSV files are supported.")

        # 4) Binary check — read first 8 KB
        try:
            with open(file_path, "rb") as f:
                head = f.read(8192)
        except OSError as e:
            return self._fail(f"Cannot read file: {e}")

        if self._is_binary(head):
            return self._fail(
                "File appears to be binary, not a valid CSV. "
                "Please upload a plain-text CSV file."
            )

        # 5) CSV parseability — try to sniff delimiter & read a few rows
        try:
            encoding = self._detect_encoding(head)
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                sample = f.read(8192)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = ","  # fall back to comma

            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                reader = csv.reader(f, delimiter=delimiter)
                header = next(reader, None)
                if not header or all(c.strip() == "" for c in header):
                    return self._fail("CSV file has no header row or is empty.")

                n_cols = len(header)

                # 6) Column count guard
                if n_cols > self.max_columns:
                    return self._fail(
                        f"CSV has {n_cols} columns (max {self.max_columns}). "
                        "Please reduce the number of columns."
                    )

                if n_cols == 0:
                    return self._fail("CSV file has no columns.")

                # Read a few data rows to verify there *is* data
                data_rows = 0
                for i, row in enumerate(reader):
                    if i >= _SNIFF_ROWS:
                        break
                    data_rows += 1

                    # 7) Formula-injection sniff
                    if self.warn_formula_injection:
                        for cell in row:
                            stripped = cell.strip()
                            if stripped and stripped[0] in ("=", "+", "-", "@", "|"):
                                warnings.append(
                                    "Some cells start with formula characters "
                                    "(=, +, -, @). This is usually harmless for "
                                    "analysis but could indicate injected formulas."
                                )
                                self.warn_formula_injection = False  # warn once
                                break

                if data_rows == 0:
                    return self._fail("CSV file has a header but no data rows.")

        except UnicodeDecodeError:
            return self._fail(
                "File contains invalid characters and could not be decoded. "
                "Please save as UTF-8 CSV and re-upload."
            )
        except Exception as e:
            logger.warning(f"CSV validation error: {e}", exc_info=True)
            return self._fail(f"File does not appear to be a valid CSV: {e}")

        result: dict = {"valid": True, "warnings": warnings}
        if warnings:
            logger.info(f"Validation passed with warnings: {warnings}")
        return result

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _fail(msg: str) -> dict:
        return {"valid": False, "error": msg, "warnings": []}

    @staticmethod
    def _is_binary(head: bytes) -> bool:
        """Return True if the first bytes look like a binary file."""
        for sig in _BINARY_SIGNATURES:
            if head.startswith(sig):
                return True
        # High ratio of null bytes → likely binary
        if len(head) > 0 and head.count(b"\x00") / len(head) > 0.05:
            return True
        return False

    @staticmethod
    def _detect_encoding(head: bytes) -> str:
        """Best-effort encoding detection from BOM or byte heuristics."""
        if head.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if head.startswith((b"\xff\xfe", b"\xfe\xff")):
            return "utf-16"
        try:
            head.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "latin-1"
