from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class SentimentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sentiment'

    def ready(self):
        """Load the sentiment model when Django starts."""
        try:
            from . import models_loader
            if models_loader.load_models():
                logger.info("Sentiment model loaded at startup.")
            else:
                logger.warning(
                    "Sentiment model could not be loaded. "
                    "Run 'python scripts/train.py' to train the model first."
                )
        except Exception as e:
            logger.error(f"Error loading sentiment model at startup: {e}")
