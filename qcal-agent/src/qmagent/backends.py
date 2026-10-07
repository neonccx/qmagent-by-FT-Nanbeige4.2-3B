"""Select an explicitly requested backend or safely discover installed hardware."""

def make_backend(settings, seed=None):
    selected_seed = settings.seed if seed is None else seed
    if settings.backend in {"auto", "hardware"}:
        from .hardware_backend import discover_hardware_backend
        backend = discover_hardware_backend()
        if backend is not None:
            return backend
        if settings.backend == "hardware":
            raise RuntimeError("Hardware mode requested, but no healthy qcal_agent.hardware provider was detected")
    if settings.backend in {"physical", "legacy"}:
        from .physical_backend import PhysicalSimulator
        return PhysicalSimulator(selected_seed, settings.noise_scale, settings.simulation_profile)
    if settings.backend == "auto":
        from .physical_backend import PhysicalSimulator
        return PhysicalSimulator(selected_seed, settings.noise_scale, settings.simulation_profile)
    raise ValueError("Unknown backend")
