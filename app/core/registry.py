import importlib
import pkgutil
import time
import yaml
from loguru import logger
from pathlib import Path
from typing import Dict, List, Type

from app.models import BaseModel


_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(*model_ids: str):
    """Register a model class with one or more model IDs.

    Args:
        *model_ids: One or more model identifier strings.

    Returns:
        Decorator function that registers the class.
    """

    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        for model_id in model_ids:
            if model_id in _MODEL_REGISTRY:
                logger.warning(
                    f"Model '{model_id}' is already registered. "
                    f"Overwriting with {cls.__name__}"
                )
            _MODEL_REGISTRY[model_id] = cls
        return cls

    return decorator


class ModelRegistry:
    """Model loader and registry."""

    def __init__(self, config_dir: Path):
        """Initialize model loader.

        Args:
            config_dir: Path to configs directory.
        """
        self.config_dir = config_dir
        self.models: Dict[str, BaseModel] = {}
        self.model_registry = self._build_registry()

    def _build_registry(self) -> Dict[str, type]:
        """Build model registry mapping model_id to implementation class.

        Note:
            Multiple model_ids can share the same implementation class.
            The model_path in config file determines which weights to load.
            Example: yolo11s, yolo11m, yolo11l can all use YOLO11nDetection.
            Automatically discovers and imports all model modules in the first level only.

        Returns:
            Dictionary mapping model_id to model class.
        """
        import app.models

        for importer, modname, ispkg in pkgutil.iter_modules(
            app.models.__path__, app.models.__name__ + "."
        ):
            if not ispkg:
                if not modname.startswith("app.models._"):
                    try:
                        importlib.import_module(modname)
                    except Exception as e:
                        logger.warning(
                            f"Failed to import model module {modname}: {e}"
                        )

        return _MODEL_REGISTRY

    def _read_models_config(self) -> Dict:
        """Read models.yaml configuration.

        Returns:
            Dictionary with enabled_models list.
        """
        models_yaml = self.config_dir / "models.yaml"
        if not models_yaml.exists():
            logger.warning(f"models.yaml not found at {models_yaml}")
            return {"enabled_models": []}

        with open(models_yaml, "r") as f:
            return yaml.safe_load(f) or {"enabled_models": []}

    def _read_model_config(self, model_id: str) -> Dict:
        """Read individual model configuration.

        Args:
            model_id: Model identifier.

        Returns:
            Model configuration dictionary.
        """
        config_path = self.config_dir / "auto_labeling" / f"{model_id}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError(f"Empty configuration file: {config_path}")

        return config

    def _validate_configs(self, configs: List[Dict]):
        """Validate model configurations.

        Args:
            configs: List of model configuration dictionaries.
        """
        model_ids = set()
        display_names = {}

        for config in configs:
            model_id = config.get("model_id")
            display_name = config.get("display_name")

            if model_id in model_ids:
                raise ValueError(f"Duplicate model_id found: {model_id}")
            model_ids.add(model_id)

            if display_name in display_names:
                logger.warning(
                    f"Duplicate display_name '{display_name}' found: "
                    f"model_id '{model_id}' and '{display_names[display_name]}'"
                )
            display_names[display_name] = model_id

            self._validate_widgets(model_id, config.get("widgets", []))

    def _validate_widgets(self, model_id: str, widgets: List[Dict]):
        """Validate widget configuration.

        Args:
            model_id: Model identifier.
            widgets: List of widget configurations.
        """
        required_defaults = {
            "edit_conf": {"type": (int, float), "range": (0.0, 1.0)},
            "edit_iou": {"type": (int, float), "range": (0.0, 1.0)},
            "mask_fineness_slider": {"type": int, "range": (1, 100)},
            "toggle_preserve_existing_annotations": {"type": bool},
        }

        widget_dict = {w["name"]: w for w in widgets}

        # (NOTE) edit_text must requires button_send
        if "edit_text" in widget_dict and "button_send" not in widget_dict:
            raise ValueError(
                f"Model [{model_id}]: Widget 'edit_text' requires 'button_send' button. "
                f"Please add 'button_send' to widgets configuration."
            )

        for widget_name, rules in required_defaults.items():
            if widget_name in widget_dict:
                widget = widget_dict[widget_name]
                value = widget.get("value")

                if value is None:
                    raise ValueError(
                        f"Model [{model_id}]: Widget '{widget_name}' requires a default value"
                    )

                if not isinstance(value, rules["type"]):
                    raise ValueError(
                        f"Model [{model_id}]: Widget '{widget_name}' value must be {rules['type']}"
                    )

                if "range" in rules:
                    min_val, max_val = rules["range"]
                    if not (min_val <= value <= max_val):
                        raise ValueError(
                            f"Model [{model_id}]: Widget '{widget_name}' value {value} "
                            f"out of range [{min_val}, {max_val}]"
                        )

    def load_all_models(self):
        """Load all enabled models on startup."""
        models_config = self._read_models_config()
        enabled = models_config.get("enabled_models", [])

        if not enabled:
            logger.warning("No models enabled in models.yaml")
            return

        logger.info(f"Loading {len(enabled)} enabled model(s)...")

        configs = []
        for model_id in enabled:
            try:
                config = self._read_model_config(model_id)
                configs.append(config)
            except Exception as e:
                logger.error(f"Failed to read config for [{model_id}]: {e}")
                continue

        if configs:
            try:
                self._validate_configs(configs)
            except Exception as e:
                logger.error(f"Configuration validation failed: {e}")

        for model_id in enabled:
            try:
                self._load_single_model(model_id)
            except Exception as e:
                logger.error(f"Failed to load model [{model_id}]: {e}")
                continue

        if len(self.models) == 0:
            logger.warning("No models were successfully loaded")
        else:
            logger.info(
                f"Successfully loaded {len(self.models)}/{len(enabled)} model(s)"
            )

    def _load_single_model(self, model_id: str):
        """Load a single model.

        Args:
            model_id: Model identifier.
        """
        config = self._read_model_config(model_id)

        if model_id not in self.model_registry:
            raise ValueError(
                f"Model '{model_id}' not registered. "
                f"Add it to _build_registry() in loader.py"
            )

        model_class = self.model_registry[model_id]
        instance = model_class(config)

        logger.info(f"Loading [{model_id}] ({config['display_name']})...")
        start_time = time.time()

        instance.load()

        elapsed = time.time() - start_time
        self.models[model_id] = instance
        logger.info(
            f"Model [{model_id}] loaded successfully (took {elapsed:.2f}s)"
        )

    def get_model(self, model_id: str) -> BaseModel:
        """Get loaded model instance.

        Args:
            model_id: Model identifier.

        Returns:
            Model instance.
        """
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not loaded")
        return self.models[model_id]

    def get_all_models_info(self) -> Dict[str, Dict]:
        """Get metadata for all loaded models.

        Returns:
            Dictionary mapping model_id to metadata.
        """
        return {
            model_id: model.get_metadata()
            for model_id, model in self.models.items()
        }

    def unload_all_models(self):
        """Unload all models and free resources."""
        logger.info("Unloading all models...")
        for model_id, model in self.models.items():
            try:
                model.unload()
                logger.info(f"Model [{model_id}] unloaded")
            except Exception as e:
                logger.error(f"Failed to unload model [{model_id}]: {e}")
        self.models.clear()
