"""FactoryRegistry — central name → class registry for YAML-driven composition."""

from typing import Any, Dict, Type

from flywheel.protocols.ooda.ooda_role import OODARole


_REGISTRY: Dict[str, Type] = {}


def register_class(name: str, cls: Type) -> None:
    """Register a class so YAML configs can refer to it by name."""
    _REGISTRY[name] = cls


def get_registered(name: str) -> Type:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown component: {name!r}. "
            f"Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


class FactoryRegistry:
    """Builds components from YAML configuration."""

    def __init__(self):
        self._registry = _REGISTRY

    def auto_register(self) -> None:
        """Import all known modules to populate the registry."""
        from flywheel.factory.auto_register import register_all
        register_all()

    def get(self, name: str) -> Type:
        return get_registered(name)

    def create(self, name: str, **kwargs) -> Any:
        return self.get(name)(**kwargs)

    def build_ooda_role(self, config: Dict[str, Any]) -> OODARole:
        return OODARole(
            observe=self.create(config["observe"]),
            orient=self.create(config["orient"]),
            decide=self.create(config["decide"]),
            act=self.create(config["act"]),
            params=config.get("params", {}),
        )

    def build_from_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        components: Dict[str, Any] = {}

        for role_name in ("redteam", "verifier", "refinement"):
            if role_name in config:
                components[role_name] = self.build_ooda_role(config[role_name])

        for comp_name in ("proposer", "oracle_adapter", "oracle",
                          "flywheel_overlay", "enforcement", "triage",
                          "blue_team", "knowledge_base"):
            if comp_name in config:
                cfg = config[comp_name]
                if isinstance(cfg, dict):
                    cls_name = cfg.get("class")
                    params = cfg.get("params", {})
                else:
                    cls_name = cfg
                    params = {}
                components[comp_name] = self.create(cls_name, **params)

        return components

    def list_registered(self) -> list:
        return sorted(self._registry.keys())
