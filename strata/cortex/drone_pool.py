"""Drone pool — auto-scaling ultra-small drone instances.

Spawns additional drone instances to parallelize chunk processing when queue
depth grows, provided enough VRAM headroom. Scales back down when idle.
"""

from __future__ import annotations

from typing import Callable, Optional

from sieve.ultra_small import UltraSmallDrone

DroneFactory = Callable[[], object]


class DronePool:
    def __init__(
        self,
        max_instances: int = 3,
        drone_factory: Optional[DroneFactory] = None,
        vram_per_instance_mb: int = 100,
    ) -> None:
        self.instances: list = []
        self.max_instances = max_instances
        self.drone_factory = drone_factory or (lambda: UltraSmallDrone())
        self.vram_per_instance_mb = vram_per_instance_mb
        # Ensure at least one instance.
        if not self.instances:
            self.instances.append(self.drone_factory())

    def scale_if_needed(self, queue_depth: int, available_vram_mb: int) -> int:
        """Scale to match load; returns the new instance count."""
        needed = min(queue_depth // 10, self.max_instances)
        vram = available_vram_mb
        while len(self.instances) < needed and vram > 500:
            self.instances.append(self.drone_factory())
            vram -= self.vram_per_instance_mb

        while len(self.instances) > 1 and queue_depth < 3:
            self.instances.pop()

        return len(self.instances)
