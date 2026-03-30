#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" CPU resource dynamic detection tool
==================
Real-time CPU idle resource detection based on psutil library, supporting intelligent parallel resource allocation """
import psutil
import time
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def get_available_cpu_cores(sample_duration: float = 1.0, cpu_threshold: float = 95.0) -> Tuple[int, float]:
    """     Dynamically detect the number of CPU cores available in the current system
    
    Args:
        sample_duration: CPU usage sampling duration (seconds)
        cpu_threshold: CPU usage threshold, cores exceeding this value are considered "busy"
        
    Returns:
        Tuple[int, float]: (number of available cores, current system average CPU usage)     """
    try:
        # Get the total number of cores in the system
        total_cores = psutil.cpu_count(logical=True)
        
        # Sampled CPU usage - counted separately by core
        cpu_percentages = psutil.cpu_percent(interval=sample_duration, percpu=True)
        
        # Calculate average system usage
        avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages)
        
        # Count the number of idle cores (cores with usage below the threshold)
        available_cores = sum(1 for usage in cpu_percentages if usage < cpu_threshold)

        
        # At least 1 core is guaranteed to be available
        available_cores = max(1, available_cores)
        
        logger.info(f"CPUResource detection results:")
        logger.info(f"  - System core: {total_cores}")
        logger.info(f"  - Average CPU usage: {avg_cpu_usage:.1f}%")
        logger.info(f"  - Number of cores available (usage<{cpu_threshold}%): {available_cores}")
        
        return available_cores, avg_cpu_usage
        
    except Exception as e:
        logger.warning(f"CPUDetection failed, use default value: {e}")
        # Downgrade plan: return 80% of the total number of cores
        fallback_cores = max(1, int(psutil.cpu_count() * 0.8))
        return fallback_cores, 0.0

def calculate_optimal_workers(target_count: int, available_cores: int, cores_per_worker: int) -> Tuple[int, int]:
    """     Calculate the optimal number of worker processes and cores per process
    
    Args:
        target_count: number of target tasks (such as the number of receptors)
        available_cores: Number of available CPU cores
        cores_per_worker: The desired number of cores per worker process
        
    Returns:
        Tuple[int, int]: (actual number of worker processes, actual number of cores per process)     """
    if cores_per_worker == -1:
        # Automatic mode: distributes all available cores evenly
        if target_count <= available_cores:
            # Plenty of cores, assign multiple cores to each task
            actual_workers = target_count
            actual_cores_per_worker = max(1, available_cores // target_count)
        else:
            # Not enough cores, each core handles one task
            actual_workers = available_cores
            actual_cores_per_worker = 1
    else:
        # Fixed mode: use specified number of cores
        max_possible_workers = available_cores // cores_per_worker
        actual_workers = min(target_count, max_possible_workers, available_cores)
        actual_cores_per_worker = cores_per_worker
    
    # Ensure there is at least 1 worker process
    actual_workers = max(1, actual_workers)
    actual_cores_per_worker = max(1, actual_cores_per_worker)
    
    logger.info(f"Parallel resource allocation results:")
    logger.info(f"  - Target number of tasks: {target_count}")
    logger.info(f"  - Actual number of worker processes: {actual_workers}")
    logger.info(f"  - Number of CPU cores per process: {actual_cores_per_worker}")
    logger.info(f"  - Total number of cores used: {actual_workers* actual_cores_per_worker}")
    
    return actual_workers, actual_cores_per_worker 