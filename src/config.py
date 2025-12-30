"""
Configuration management for GPU Collision Detection system.

This module provides a centralized configuration system with support for:
- Default configuration values
- Platform-specific settings
- Environment variable overrides
- Configuration file loading (YAML/JSON)

Platform Support:
    - Linux: Full support
    - Windows: Full support
    - macOS: Limited support (depends on CUDA/OpenGL availability)
"""

import os
import platform
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Configuration manager for simulation parameters.
    
    Provides a hierarchical configuration system:
    1. Default values (defined in this class)
    2. Platform-specific overrides
    3. User configuration file (if present)
    4. Environment variable overrides
    5. Runtime parameter overrides
    
    Usage:
        >>> config = Config()
        >>> config.get('simulation.num_objects', default=1000)
        1000
        >>> config.set('simulation.num_objects', 2000)
        >>> config.save('my_config.json')
    """
    
    # Default configuration values
    DEFAULTS = {
        # Simulation parameters
        'simulation': {
            'num_objects': 1000,
            'world_bounds': [[-50, 0, -50], [50, 50, 50]],
            'cell_size': 2.0,
            'dt': 1.0 / 60.0,
            'gravity': [0.0, -9.81, 0.0],
            'damping': 0.01,
        },
        
        # GPU parameters
        'gpu': {
            'device_id': 0,
            'block_size': 256,
            'max_collision_pairs': 500000,
            'enable_profiling': False,
        },
        
        # Visualization parameters
        'visualization': {
            'enabled': True,
            'width': 1920,
            'height': 1080,
            'title': 'GPU Collision Detection',
            'fps_target': 60,
            'show_stats': True,
        },
        
        # Recording parameters
        'recording': {
            'enabled': False,
            'output_dir': 'output',
            'filename': 'simulation.mp4',
            'codec': 'h264',
            'bitrate': '5M',
            'fps': 60,
        },
        
        # Performance parameters
        'performance': {
            'enable_monitoring': True,
            'log_interval': 60,  # frames
            'export_csv': False,
            'csv_path': 'output/performance.csv',
        },
        
        # Platform-specific parameters
        'platform': {
            'auto_detect': True,
            'fallback_to_cpu': False,  # Not implemented yet
        }
    }
    
    # Platform-specific overrides
    PLATFORM_OVERRIDES = {
        'Windows': {
            'recording': {
                'codec': 'h264',  # Works well on Windows
            }
        },
        'Linux': {
            'recording': {
                'codec': 'h264',
            }
        },
        'Darwin': {  # macOS
            'recording': {
                'codec': 'h264',
            },
            'visualization': {
                'width': 1280,  # Lower default for macOS
                'height': 720,
            }
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file (JSON or YAML)
        """
        self._config = self._deep_copy(self.DEFAULTS)
        self._apply_platform_overrides()
        
        if config_file and os.path.exists(config_file):
            self.load(config_file)
        
        self._apply_env_overrides()
    
    def _deep_copy(self, d: Dict) -> Dict:
        """Deep copy dictionary to avoid reference issues."""
        if isinstance(d, dict):
            return {k: self._deep_copy(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._deep_copy(item) for item in d]
        else:
            return d
    
    def _apply_platform_overrides(self):
        """Apply platform-specific configuration overrides."""
        system = platform.system()
        if system in self.PLATFORM_OVERRIDES:
            overrides = self.PLATFORM_OVERRIDES[system]
            self._merge_config(overrides)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # GPU device selection
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            try:
                device_id = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
                self.set('gpu.device_id', device_id)
            except (ValueError, IndexError):
                pass
        
        # Output directory
        if 'OUTPUT_DIR' in os.environ:
            self.set('recording.output_dir', os.environ['OUTPUT_DIR'])
        
        # Number of objects (for batch processing)
        if 'NUM_OBJECTS' in os.environ:
            try:
                num_objects = int(os.environ['NUM_OBJECTS'])
                self.set('simulation.num_objects', num_objects)
            except ValueError:
                pass
    
    def _merge_config(self, overrides: Dict, target: Optional[Dict] = None):
        """
        Recursively merge override configuration into target.
        
        Args:
            overrides: Configuration overrides to apply
            target: Target configuration (uses self._config if None)
        """
        if target is None:
            target = self._config
        
        for key, value in overrides.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(value, target[key])
            else:
                target[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated key (e.g., 'simulation.num_objects')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Examples:
            >>> config.get('simulation.num_objects')
            1000
            >>> config.get('nonexistent.key', default=42)
            42
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by dot-separated key.
        
        Args:
            key: Dot-separated key (e.g., 'simulation.num_objects')
            value: Value to set
        
        Examples:
            >>> config.set('simulation.num_objects', 2000)
            >>> config.set('gpu.device_id', 1)
        """
        keys = key.split('.')
        target = self._config
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
    
    def load(self, filepath: str):
        """
        Load configuration from file.
        
        Supports JSON and YAML formats (auto-detected by extension).
        
        Args:
            filepath: Path to configuration file
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if path.suffix == '.json':
            with open(filepath, 'r') as f:
                loaded_config = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(filepath, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            except ImportError:
                raise ValueError("PyYAML not installed. Install with: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        
        self._merge_config(loaded_config)
    
    def save(self, filepath: str):
        """
        Save current configuration to file.
        
        Format is determined by file extension (.json or .yaml/.yml).
        
        Args:
            filepath: Path to save configuration file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(self._config, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False)
            except ImportError:
                raise ValueError("PyYAML not installed. Install with: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
    
    def get_all(self) -> Dict:
        """
        Get all configuration as a dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self._deep_copy(self._config)
    
    def print_config(self):
        """Print current configuration in a readable format."""
        self._print_dict(self._config)
    
    def _print_dict(self, d: Dict, indent: int = 0):
        """Recursively print dictionary with indentation."""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    def detect_gpu(self) -> Dict[str, Any]:
        """
        Detect GPU capabilities and return information.
        
        Returns:
            Dictionary with GPU information:
            - available: bool
            - device_name: str
            - compute_capability: tuple
            - total_memory: int (bytes)
            - cuda_version: str
        """
        gpu_info = {
            'available': False,
            'device_name': 'None',
            'compute_capability': (0, 0),
            'total_memory': 0,
            'cuda_version': 'N/A',
            'driver_version': 'N/A'
        }
        
        try:
            import cupy as cp
            
            if cp.cuda.is_available():
                device = cp.cuda.Device(self.get('gpu.device_id', 0))
                
                gpu_info['available'] = True
                gpu_info['device_name'] = device.name.decode('utf-8') if isinstance(device.name, bytes) else device.name
                gpu_info['compute_capability'] = device.compute_capability
                gpu_info['total_memory'] = device.mem_info[1]  # Total memory in bytes
                gpu_info['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()
                gpu_info['driver_version'] = cp.cuda.runtime.driverGetVersion()
        
        except Exception as e:
            print(f"Warning: Could not detect GPU: {e}")
        
        return gpu_info
    
    def get_platform_info(self) -> Dict[str, str]:
        """
        Get platform information.
        
        Returns:
            Dictionary with platform details:
            - system: Operating system (Windows/Linux/Darwin)
            - release: OS release version
            - machine: Machine architecture
            - processor: Processor name
            - python_version: Python version
        """
        return {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }


# Global configuration instance (singleton pattern)
_global_config = None


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton).
    
    Args:
        config_file: Optional configuration file to load
    
    Returns:
        Global Config instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = Config(config_file)
    
    return _global_config


def reset_config():
    """Reset global configuration to defaults."""
    global _global_config
    _global_config = None
