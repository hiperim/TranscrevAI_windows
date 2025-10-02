"""
PLATFORM MANAGER - BASE MULTI-PLATFORM
Detecção automática de plataforma e otimizações específicas
Preparado para Windows, Linux, macOS, Android, iOS
"""

import platform
import logging
import os
import sys
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class PlatformType(Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    ANDROID = "android"
    IOS = "ios"
    UNKNOWN = "unknown"

class DeviceType(Enum):
    DESKTOP = "desktop"
    LAPTOP = "laptop"
    WORKSTATION = "workstation"
    SERVER = "server"
    MOBILE = "mobile"
    TABLET = "tablet"
    EMBEDDED = "embedded"
    UNKNOWN = "unknown"

@dataclass
class PlatformCapabilities:
    """Capacidades específicas da plataforma"""
    supports_gpu_acceleration: bool
    supports_multithreading: bool
    supports_memory_mapping: bool
    supports_hardware_decode: bool
    max_memory_usage_percent: float
    battery_optimized: bool
    thermal_limited: bool
    file_system_case_sensitive: bool

class PlatformManager:
    """Gerenciador de plataforma com otimizações específicas"""

    def __init__(self):
        self.platform_type = self._detect_platform()
        self.device_type = self._detect_device_type()
        self.is_notebook = self._detect_notebook()
        self.capabilities = self._get_platform_capabilities()

        logger.info(f"Platform detected: {self.platform_type.value}")
        logger.info(f"Device type: {self.device_type.value}")
        logger.info(f"Is notebook: {self.is_notebook}")

    def _detect_platform(self) -> PlatformType:
        """Detectar plataforma atual"""
        system = platform.system().lower()

        if system == "windows":
            return PlatformType.WINDOWS
        elif system == "linux":
            # Verificar se é Android (Linux-based)
            if "android" in platform.platform().lower():
                return PlatformType.ANDROID
            return PlatformType.LINUX
        elif system == "darwin":
            # Detectar iOS vs macOS
            if "iphone" in platform.platform().lower() or "ipad" in platform.platform().lower():
                return PlatformType.IOS
            return PlatformType.MACOS
        else:
            return PlatformType.UNKNOWN

    def _detect_device_type(self) -> DeviceType:
        """Detectar tipo de dispositivo"""
        try:
            # Mobile platforms
            if self.platform_type in [PlatformType.ANDROID, PlatformType.IOS]:
                return DeviceType.MOBILE

            # Desktop platforms
            if self.platform_type == PlatformType.WINDOWS:
                return self._detect_windows_device_type()
            elif self.platform_type == PlatformType.MACOS:
                return self._detect_macos_device_type()
            elif self.platform_type == PlatformType.LINUX:
                return self._detect_linux_device_type()

            return DeviceType.UNKNOWN

        except Exception as e:
            logger.warning(f"Device type detection failed: {e}")
            return DeviceType.UNKNOWN

    def _detect_windows_device_type(self) -> DeviceType:
        """Detectar tipo de dispositivo Windows"""
        try:
            import psutil

            # Verificar se há bateria (indica laptop)
            battery = psutil.sensors_battery()
            if battery is not None:
                return DeviceType.LAPTOP

            # Verificar quantidade de RAM e cores para classificar
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)

            if memory_gb >= 32 and cpu_cores and cpu_cores >= 8:
                return DeviceType.WORKSTATION
            elif memory_gb >= 16 and cpu_cores and cpu_cores >= 6:
                return DeviceType.DESKTOP
            else:
                # Hardware limitado pode ser laptop sem bateria detectada
                return DeviceType.LAPTOP

        except Exception as e:
            logger.warning(f"Windows device detection failed: {e}")
            return DeviceType.DESKTOP

    def _detect_macos_device_type(self) -> DeviceType:
        """Detectar tipo de dispositivo macOS"""
        try:
            # Usar system_profiler ou sysctl para detectar modelo
            machine = platform.machine().lower()

            if "macbook" in platform.platform().lower():
                return DeviceType.LAPTOP
            elif "imac" in platform.platform().lower():
                return DeviceType.DESKTOP
            elif "mac" in machine and ("arm" in machine or "m1" in machine or "m2" in machine):
                # Apple Silicon - geralmente laptops ou iMacs
                return DeviceType.LAPTOP  # Assumir laptop para otimizações conservadoras
            else:
                return DeviceType.DESKTOP

        except Exception as e:
            logger.warning(f"macOS device detection failed: {e}")
            return DeviceType.DESKTOP

    def _detect_linux_device_type(self) -> DeviceType:
        """Detectar tipo de dispositivo Linux"""
        try:
            import psutil

            # Verificar DMI (Desktop Management Interface)
            try:
                with open('/sys/class/dmi/id/chassis_type', 'r') as f:
                    chassis_type = int(f.read().strip())

                # Chassis types: 9=Laptop, 10=Notebook, 14=Sub Notebook
                if chassis_type in [9, 10, 14]:
                    return DeviceType.LAPTOP
                elif chassis_type in [3, 4, 6, 7]:  # Desktop, Low Profile Desktop, Mini Tower, Tower
                    return DeviceType.DESKTOP
                elif chassis_type in [17, 23]:  # Server, Rack Mount
                    return DeviceType.SERVER
            except:
                pass

            # Fallback: verificar bateria
            battery = psutil.sensors_battery()
            if battery is not None:
                return DeviceType.LAPTOP

            return DeviceType.DESKTOP

        except Exception as e:
            logger.warning(f"Linux device detection failed: {e}")
            return DeviceType.DESKTOP

    def _detect_notebook(self) -> bool:
        """Detectar se é especificamente um notebook/laptop"""
        return self.device_type in [DeviceType.LAPTOP, DeviceType.MOBILE, DeviceType.TABLET]

    def _get_platform_capabilities(self) -> PlatformCapabilities:
        """Obter capacidades específicas da plataforma"""
        if self.platform_type == PlatformType.WINDOWS:
            return PlatformCapabilities(
                supports_gpu_acceleration=True,
                supports_multithreading=True,
                supports_memory_mapping=True,
                supports_hardware_decode=True,
                max_memory_usage_percent=0.75 if self.is_notebook else 0.85,
                battery_optimized=self.is_notebook,
                thermal_limited=self.is_notebook,
                file_system_case_sensitive=False
            )
        elif self.platform_type == PlatformType.MACOS:
            return PlatformCapabilities(
                supports_gpu_acceleration=True,  # Metal Performance Shaders
                supports_multithreading=True,
                supports_memory_mapping=True,
                supports_hardware_decode=True,  # VideoToolbox
                max_memory_usage_percent=0.70 if self.is_notebook else 0.80,
                battery_optimized=self.is_notebook,
                thermal_limited=True,  # Macs são sempre thermal-limited
                file_system_case_sensitive=True
            )
        elif self.platform_type == PlatformType.LINUX:
            return PlatformCapabilities(
                supports_gpu_acceleration=True,  # CUDA, OpenCL, Vulkan
                supports_multithreading=True,
                supports_memory_mapping=True,
                supports_hardware_decode=True,  # VAAPI, VDPAU
                max_memory_usage_percent=0.80 if self.is_notebook else 0.90,
                battery_optimized=self.is_notebook,
                thermal_limited=self.is_notebook,
                file_system_case_sensitive=True
            )
        elif self.platform_type in [PlatformType.ANDROID, PlatformType.IOS]:
            return PlatformCapabilities(
                supports_gpu_acceleration=True,  # Mobile GPUs
                supports_multithreading=True,
                supports_memory_mapping=False,  # Limitado em mobile
                supports_hardware_decode=True,  # Hardware codecs
                max_memory_usage_percent=0.60,  # Muito conservador em mobile
                battery_optimized=True,
                thermal_limited=True,
                file_system_case_sensitive=self.platform_type == PlatformType.IOS
            )
        else:
            return PlatformCapabilities(
                supports_gpu_acceleration=False,
                supports_multithreading=True,
                supports_memory_mapping=False,
                supports_hardware_decode=False,
                max_memory_usage_percent=0.50,
                battery_optimized=True,
                thermal_limited=True,
                file_system_case_sensitive=True
            )

    def get_notebook_optimizations(self) -> Dict[str, Any]:
        """Otimizações específicas para notebooks"""
        if not self.is_notebook:
            return {}

        return {
            'power_management': {
                'enable_cpu_throttling': True,
                'reduce_gpu_clock': True,
                'adaptive_performance': True,
                'battery_saver_mode': True
            },
            'thermal_management': {
                'temperature_monitoring': True,
                'thermal_throttling': True,
                'fan_curve_optimization': True,
                'cpu_temp_limit': 75  # Celsius
            },
            'memory_management': {
                'aggressive_cleanup': True,
                'reduced_cache_size': True,
                'memory_pressure_sensitive': True,
                'swap_avoidance': True
            },
            'performance_profiles': {
                'balanced_mode': True,
                'eco_mode': True,
                'performance_on_ac': True,
                'power_save_on_battery': True
            }
        }

    def get_platform_specific_paths(self) -> Dict[str, str]:
        """Caminhos específicos da plataforma"""
        if self.platform_type == PlatformType.WINDOWS:
            return {
                'cache_dir': os.path.expandvars(r'%LOCALAPPDATA%\TranscrevAI\cache'),
                'models_dir': os.path.expandvars(r'%LOCALAPPDATA%\TranscrevAI\models'),
                'temp_dir': os.path.expandvars(r'%TEMP%\TranscrevAI'),
                'logs_dir': os.path.expandvars(r'%LOCALAPPDATA%\TranscrevAI\logs')
            }
        elif self.platform_type == PlatformType.MACOS:
            home = os.path.expanduser('~')
            return {
                'cache_dir': f'{home}/Library/Caches/TranscrevAI',
                'models_dir': f'{home}/Library/Application Support/TranscrevAI/models',
                'temp_dir': '/tmp/TranscrevAI',
                'logs_dir': f'{home}/Library/Logs/TranscrevAI'
            }
        elif self.platform_type == PlatformType.LINUX:
            home = os.path.expanduser('~')
            return {
                'cache_dir': f'{home}/.cache/transcrevai',
                'models_dir': f'{home}/.local/share/transcrevai/models',
                'temp_dir': '/tmp/transcrevai',
                'logs_dir': f'{home}/.local/share/transcrevai/logs'
            }
        elif self.platform_type in [PlatformType.ANDROID, PlatformType.IOS]:
            # Para mobile, usar diretórios relativos
            return {
                'cache_dir': './cache',
                'models_dir': './models',
                'temp_dir': './temp',
                'logs_dir': './logs'
            }
        else:
            return {
                'cache_dir': './cache',
                'models_dir': './models',
                'temp_dir': './temp',
                'logs_dir': './logs'
            }

    def get_optimal_model_loading_strategy(self) -> Dict[str, Any]:
        """Estratégia de carregamento otimizada para a plataforma"""
        base_strategy = {
            'sequential_loading': self.is_notebook,
            'memory_mapped_files': self.capabilities.supports_memory_mapping,
            'use_hardware_decode': self.capabilities.supports_hardware_decode,
            'enable_gpu_acceleration': self.capabilities.supports_gpu_acceleration,
            'max_memory_percent': self.capabilities.max_memory_usage_percent,
            'thermal_aware': self.capabilities.thermal_limited,
            'battery_optimized': self.capabilities.battery_optimized
        }

        # Otimizações específicas para notebooks
        if self.is_notebook:
            base_strategy.update({
                'preload_encoder_only': True,  # Carregar decoder on-demand
                'aggressive_memory_cleanup': True,
                'reduce_model_precision': True,  # FP16 em vez de FP32
                'enable_model_quantization': True,
                'chunked_processing': True,
                'background_loading_priority': 'low'
            })

        return base_strategy

    def get_websocket_config(self) -> Dict[str, Any]:
        """Configuração WebSocket otimizada para plataforma"""
        base_config = {
            'message_compression': True,
            'heartbeat_interval': 30,
            'max_connections': 10 if not self.is_notebook else 5,
            'buffer_size': 65536 if not self.is_notebook else 32768
        }

        # Mobile/notebook optimizations
        if self.is_notebook or self.platform_type in [PlatformType.ANDROID, PlatformType.IOS]:
            base_config.update({
                'low_latency_mode': False,
                'aggressive_throttling': True,
                'connection_pooling': True,
                'background_disconnect': True  # Desconectar em background
            })

        return base_config

    def supports_feature(self, feature: str) -> bool:
        """Verificar se plataforma suporta feature específica"""
        feature_map = {
            'gpu_acceleration': self.capabilities.supports_gpu_acceleration,
            'multithreading': self.capabilities.supports_multithreading,
            'memory_mapping': self.capabilities.supports_memory_mapping,
            'hardware_decode': self.capabilities.supports_hardware_decode,
            'background_processing': self.platform_type != PlatformType.IOS,  # iOS limits background
            'file_watching': self.platform_type != PlatformType.ANDROID,
            'system_notifications': True,  # Todas as plataformas suportam
            'clipboard_access': self.platform_type in [PlatformType.WINDOWS, PlatformType.MACOS, PlatformType.LINUX]
        }

        return feature_map.get(feature, False)

    def get_status(self) -> Dict[str, Any]:
        """Status da plataforma"""
        return {
            'platform': {
                'type': self.platform_type.value,
                'device_type': self.device_type.value,
                'is_notebook': self.is_notebook,
                'system': platform.system(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'capabilities': {
                'gpu_acceleration': self.capabilities.supports_gpu_acceleration,
                'multithreading': self.capabilities.supports_multithreading,
                'memory_mapping': self.capabilities.supports_memory_mapping,
                'hardware_decode': self.capabilities.supports_hardware_decode,
                'max_memory_percent': self.capabilities.max_memory_usage_percent,
                'battery_optimized': self.capabilities.battery_optimized,
                'thermal_limited': self.capabilities.thermal_limited
            },
            'optimizations': self.get_notebook_optimizations() if self.is_notebook else {},
            'paths': self.get_platform_specific_paths()
        }

# Instância global
platform_manager = PlatformManager()