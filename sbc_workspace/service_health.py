import os
import time
import json
from pathlib import Path

def get_system_health():
    """
    Mengambil metrik kondisi kesehatan Raspberry Pi 5:
    - Suhu CPU (°C)
    - Penggunaan CPU (%)
    - Penggunaan RAM (%)
    - Status Disk
    """
    health = {
        "timestamp": time.time(),
        "temperature_c": None,
        "cpu_usage_percent": None,
        "ram_usage_percent": None,
        "status": "healthy"
    }

    # 1. Baca Suhu Raspberry Pi (/sys/class/thermal/thermal_zone0/temp)
    temp_file = Path("/sys/class/thermal/thermal_zone0/temp")
    if temp_file.exists():
        try:
            with open(temp_file, "r") as f:
                health["temperature_c"] = round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            pass

    # 2. Baca Memori & CPU via psutil jika tersedia
    try:
        import psutil
        health["cpu_usage_percent"] = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        health["ram_usage_percent"] = ram.percent
        health["ram_used_mb"] = round(ram.used / (1024 * 1024), 1)
        health["ram_total_mb"] = round(ram.total / (1024 * 1024), 1)
    except ImportError:
        pass

    # Status evaluation
    if health["temperature_c"] and health["temperature_c"] > 75.0:
        health["status"] = "thermal_throttling_risk"
    elif health["ram_usage_percent"] and health["ram_usage_percent"] > 90.0:
        health["status"] = "high_memory_warning"

    return health

if __name__ == "__main__":
    info = get_system_health()
    print(json.dumps(info, indent=2))
