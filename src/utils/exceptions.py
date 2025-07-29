"""Custom exceptions for the monkey recognition system."""

from typing import Optional, Any


class MonkeyRecognitionError(Exception):
    """Base exception for monkey recognition system."""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Any] = None):
        """Initialize exception.

        Args:
            message: Error message.
            error_code: Optional error code for categorization.
            details: Optional additional details about the error.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details

    def __str__(self) -> str:
        """String representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataError(MonkeyRecognitionError):
    """Raised when data loading/processing fails."""
    pass


class ValidationError(MonkeyRecognitionError):
    """Raised when input validation fails."""
    pass


class ModelError(MonkeyRecognitionError):
    """Raised when model operations fail."""
    pass


class DetectionError(MonkeyRecognitionError):
    """Raised when face detection fails."""
    pass


class RecognitionError(MonkeyRecognitionError):
    """Raised when face recognition fails."""
    pass


class DatabaseError(MonkeyRecognitionError):
    """Raised when database operations fail."""
    pass


class ConfigurationError(MonkeyRecognitionError):
    """Raised when configuration is invalid."""
    pass


class TrainingError(MonkeyRecognitionError):
    """Raised when training fails."""
    pass


class InferenceError(MonkeyRecognitionError):
    """Raised when inference fails."""
    pass


class FileSystemError(MonkeyRecognitionError):
    """Raised when file system operations fail."""
    pass


class NetworkError(MonkeyRecognitionError):
    """Raised when network operations fail."""
    pass


class ResourceError(MonkeyRecognitionError):
    """Raised when system resources are insufficient."""
    pass


# Error codes for categorization
class ErrorCodes:
    """Standard error codes for the system."""

    # Data errors
    DATA_NOT_FOUND = "DATA_001"
    DATA_CORRUPTED = "DATA_002"
    DATA_FORMAT_INVALID = "DATA_003"
    DATA_INSUFFICIENT = "DATA_004"

    # Validation errors
    VALIDATION_FAILED = "VAL_001"
    INPUT_INVALID = "VAL_002"
    PARAMETER_INVALID = "VAL_003"
    CONSTRAINT_VIOLATED = "VAL_004"

    # Model errors
    MODEL_NOT_FOUND = "MODEL_001"
    MODEL_LOAD_FAILED = "MODEL_002"
    MODEL_CORRUPTED = "MODEL_003"
    MODEL_INCOMPATIBLE = "MODEL_004"

    # Detection errors
    DETECTION_FAILED = "DET_001"
    NO_FACES_DETECTED = "DET_002"
    DETECTION_TIMEOUT = "DET_003"

    # Recognition errors
    RECOGNITION_FAILED = "REC_001"
    FEATURE_EXTRACTION_FAILED = "REC_002"
    SIMILARITY_CALCULATION_FAILED = "REC_003"
    UNKNOWN_MONKEY = "REC_004"

    # Database errors
    DATABASE_CONNECTION_FAILED = "DB_001"
    DATABASE_CORRUPTED = "DB_002"
    DATABASE_FULL = "DB_003"
    FEATURE_NOT_FOUND = "DB_004"

    # Configuration errors
    CONFIG_NOT_FOUND = "CFG_001"
    CONFIG_INVALID = "CFG_002"
    CONFIG_MISSING_REQUIRED = "CFG_003"

    # Training errors
    TRAINING_FAILED = "TRN_001"
    TRAINING_INTERRUPTED = "TRN_002"
    TRAINING_DATA_INSUFFICIENT = "TRN_003"
    CONVERGENCE_FAILED = "TRN_004"

    # Inference errors
    INFERENCE_FAILED = "INF_001"
    INFERENCE_TIMEOUT = "INF_002"
    BATCH_SIZE_EXCEEDED = "INF_003"

    # File system errors
    FILE_NOT_FOUND = "FS_001"
    FILE_PERMISSION_DENIED = "FS_002"
    DISK_SPACE_INSUFFICIENT = "FS_003"
    FILE_CORRUPTED = "FS_004"

    # Network errors
    NETWORK_TIMEOUT = "NET_001"
    NETWORK_CONNECTION_FAILED = "NET_002"
    DOWNLOAD_FAILED = "NET_003"

    # Resource errors
    MEMORY_INSUFFICIENT = "RES_001"
    GPU_NOT_AVAILABLE = "RES_002"
    CPU_OVERLOAD = "RES_003"
    STORAGE_FULL = "RES_004"


def create_error(
    error_type: type,
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Any] = None,
    cause: Optional[Exception] = None
) -> MonkeyRecognitionError:
    """Create a standardized error with optional chaining.

    Args:
        error_type: Type of error to create.
        message: Error message.
        error_code: Optional error code.
        details: Optional additional details.
        cause: Optional underlying exception.

    Returns:
        Created error instance.
    """
    error = error_type(message, error_code, details)

    if cause:
        error.__cause__ = cause

    return error