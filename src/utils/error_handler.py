"""Error handling utilities and decorators."""

import functools
import traceback
import sys
from typing import Any, Callable, Dict, List, Optional, Type, Union
import logging

from .exceptions import MonkeyRecognitionError, ErrorCodes
from .logging import get_logger


class ErrorHandler:
    """Centralized error handling for the monkey recognition system."""

    def __init__(self, logger_name: str = "error_handler"):
        """Initialize error handler.

        Args:
            logger_name: Name for the logger.
        """
        self.logger = get_logger(logger_name)
        self.error_counts = {}
        self.error_history = []

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        reraise: bool = True
    ) -> Optional[Exception]:
        """Handle an error with logging and optional recovery.

        Args:
            error: The exception that occurred.
            context: Optional context information.
            reraise: Whether to reraise the exception.

        Returns:
            The handled exception if not reraised.
        """
        # Extract error information
        error_type = type(error).__name__
        error_message = str(error)

        # Get error code if available
        error_code = getattr(error, 'error_code', None)

        # Update error statistics
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Create error record
        error_record = {
            'type': error_type,
            'message': error_message,
            'code': error_code,
            'context': context or {},
            'traceback': traceback.format_exc()
        }

        self.error_history.append(error_record)

        # Log the error
        log_message = f"[{error_code}] {error_message}" if error_code else error_message

        if context:
            log_message += f" | Context: {context}"

        if isinstance(error, MonkeyRecognitionError):
            self.logger.error(log_message)
        else:
            self.logger.error(f"Unexpected error: {log_message}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")

        if reraise:
            raise error

        return error

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics.

        Returns:
            Dictionary with error statistics.
        """
        total_errors = sum(self.error_counts.values())

        return {
            'total_errors': total_errors,
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }

    def clear_history(self) -> None:
        """Clear error history and statistics."""
        self.error_counts.clear()
        self.error_history.clear()


# Global error handler instance
_global_error_handler = ErrorHandler()


def handle_errors(
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
    context_func: Optional[Callable] = None
):
    """Decorator for handling errors in functions.

    Args:
        exceptions: Exception type(s) to catch.
        default_return: Default return value on error.
        log_errors: Whether to log errors.
        reraise: Whether to reraise exceptions after handling.
        context_func: Function to generate context information.
    """
    if not isinstance(exceptions, (list, tuple)):
        exceptions = [exceptions]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except tuple(exceptions) as e:
                context = {}

                if context_func:
                    try:
                        context = context_func(*args, **kwargs)
                    except Exception:
                        context = {'context_generation_failed': True}

                context.update({
                    'function': func.__name__,
                    'args': str(args)[:200],  # Truncate long args
                    'kwargs': str(kwargs)[:200]
                })

                if log_errors:
                    _global_error_handler.handle_error(e, context, reraise=False)

                if reraise:
                    raise

                return default_return

        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """Safely execute a function with error handling.

    Args:
        func: Function to execute.
        *args: Function arguments.
        default_return: Default return value on error.
        log_errors: Whether to log errors.
        **kwargs: Function keyword arguments.

    Returns:
        Function result or default_return on error.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        context = {
            'function': func.__name__,
            'args': str(args)[:200],
            'kwargs': str(kwargs)[:200]
        }

        if log_errors:
            _global_error_handler.handle_error(e, context, reraise=False)

        return default_return


def validate_and_handle(
    validation_func: Callable,
    value: Any,
    error_message: str = "Validation failed",
    error_code: Optional[str] = None
) -> Any:
    """Validate a value and handle errors.

    Args:
        validation_func: Function to validate the value.
        value: Value to validate.
        error_message: Error message if validation fails.
        error_code: Optional error code.

    Returns:
        Validated value.

    Raises:
        ValidationError: If validation fails.
    """
    try:
        return validation_func(value)
    except Exception as e:
        from .exceptions import ValidationError

        if isinstance(e, MonkeyRecognitionError):
            raise

        raise ValidationError(
            f"{error_message}: {str(e)}",
            error_code or ErrorCodes.VALIDATION_FAILED
        ) from e


class RetryHandler:
    """Handler for retrying operations with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ):
        """Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts.
            base_delay: Base delay between retries in seconds.
            max_delay: Maximum delay between retries.
            backoff_factor: Exponential backoff factor.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.logger = get_logger("retry_handler")

    def retry(
        self,
        func: Callable,
        *args,
        exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
        **kwargs
    ) -> Any:
        """Retry a function with exponential backoff.

        Args:
            func: Function to retry.
            *args: Function arguments.
            exceptions: Exception types to retry on.
            **kwargs: Function keyword arguments.

        Returns:
            Function result.

        Raises:
            Exception: If all retry attempts fail.
        """
        if not isinstance(exceptions, (list, tuple)):
            exceptions = [exceptions]

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except tuple(exceptions) as e:
                last_exception = e

                if attempt == self.max_retries:
                    self.logger.error(f"All retry attempts failed for {func.__name__}")
                    raise

                delay = min(
                    self.base_delay * (self.backoff_factor ** attempt),
                    self.max_delay
                )

                self.logger.warning(
                    f"Attempt {attempt + 1} failed for {func.__name__}, "
                    f"retrying in {delay:.1f}s: {str(e)}"
                )

                import time
                time.sleep(delay)

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception


def retry_on_error(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception
):
    """Decorator for retrying functions on error.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries.
        exceptions: Exception types to retry on.
    """
    retry_handler = RetryHandler(max_retries=max_retries, base_delay=base_delay)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return retry_handler.retry(func, *args, exceptions=exceptions, **kwargs)
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for handling cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit.
            recovery_timeout: Time to wait before attempting recovery.
            expected_exception: Exception type that triggers the circuit breaker.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

        self.logger = get_logger("circuit_breaker")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker.

        Args:
            func: Function to call.
            *args: Function arguments.
            **kwargs: Function keyword arguments.

        Returns:
            Function result.

        Raises:
            Exception: If circuit is open or function fails.
        """
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                self.logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                from .exceptions import ResourceError
                raise ResourceError(
                    "Circuit breaker is OPEN",
                    ErrorCodes.RESOURCE_ERROR
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        import time
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self) -> None:
        """Handle successful function call."""
        self.failure_count = 0
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.logger.info("Circuit breaker reset to CLOSED state")

    def _on_failure(self) -> None:
        """Handle failed function call."""
        import time

        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
):
    """Decorator for circuit breaker pattern.

    Args:
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Time to wait before attempting recovery.
        expected_exception: Exception type that triggers the circuit breaker.
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def get_error_statistics() -> Dict[str, Any]:
    """Get global error statistics.

    Returns:
        Dictionary with error statistics.
    """
    return _global_error_handler.get_error_statistics()


def clear_error_history() -> None:
    """Clear global error history."""
    _global_error_handler.clear_history()