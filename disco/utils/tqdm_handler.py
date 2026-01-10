# tqdm_handler.py
import logging
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    """
    A custom logging handler that allows log messages to be properly displayed
    alongside tqdm progress bars without interfering with them.

    This handler uses the `tqdm.write()` method to display log messages, ensuring
    that they are correctly displayed and do not cause tqdm progress bars to break
    or become unreadable.

    Example usage:

    ```
    import logging
    from tqdm_handler import TqdmLoggingHandler

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    handler = TqdmLoggingHandler()
    logger.addHandler(handler)
    ```

    With this setup, any log messages from the logger will be displayed using
    `tqdm.write()` and will not interfere with any active tqdm progress bars.
    """
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)
