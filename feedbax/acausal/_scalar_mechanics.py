"""Shared construction of scalar mechanical equation descriptors."""

from dataclasses import dataclass

from feedbax.acausal.base import AcausalElement, AcausalEquation, AcausalPort, Domain


@dataclass(frozen=True)
class _ScalarMechanics:
    domain: Domain
    coordinate: str
    velocity: str
    effort: str
    inertia_key: str

    def port(self, name: str) -> AcausalPort:
        return AcausalPort(
            name=name,
            domain=self.domain,
            across_vars=(self.coordinate, self.velocity),
            through_var=self.effort,
        )

    def one_port(self, element: AcausalElement, role: str) -> None:
        element.ports["flange"] = self.port("flange")
        element.element_type = role

    def sensor(self, element: AcausalElement, slot: str) -> None:
        self.one_port(element, "sensor")
        element.sensor_output = ("flange", slot)

    def inertia(self, element: AcausalElement, value: float) -> None:
        self.one_port(element, "mass")
        element.params[f"{element.name}.{self.inertia_key}"] = value
        coordinate = f"{element.name}.flange.{self.coordinate}"
        velocity = f"{element.name}.flange.{self.velocity}"
        element.equations.extend((
            AcausalEquation(
                lhs_var=coordinate,
                rhs_fn=lambda vals: vals[velocity],
                depends_on=(velocity,),
            ),
            AcausalEquation(lhs_var=velocity, rhs_fn=None, depends_on=()),
        ))

    def coupling(
        self, element: AcausalElement, parameter: str, value: float, slot: str,
    ) -> None:
        for name in ("flange_a", "flange_b"):
            element.ports[name] = self.port(name)
        key = f"{element.name}.{parameter}"
        element.params[key] = value
        across_a = f"{element.name}.flange_a.{slot}"
        across_b = f"{element.name}.flange_b.{slot}"
        effort_a = f"{element.name}.flange_a.{self.effort}"
        effort_b = f"{element.name}.flange_b.{self.effort}"
        element.equations.extend((
            AcausalEquation(
                lhs_var=effort_b,
                rhs_fn=lambda vals: vals[key] * (vals[across_a] - vals[across_b]),
                depends_on=(across_a, across_b),
                is_through_def=True,
            ),
            AcausalEquation(
                lhs_var=effort_a,
                rhs_fn=lambda vals: -vals[effort_b],
                depends_on=(effort_b,),
                is_through_def=True,
            ),
        ))


TRANSLATIONAL = _ScalarMechanics(Domain.TRANSLATIONAL, "pos", "vel", "force", "mass")
ROTATIONAL = _ScalarMechanics(
    Domain.ROTATIONAL, "angle", "angular_vel", "torque", "inertia",
)
