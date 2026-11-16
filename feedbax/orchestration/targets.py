"""Resolve named profiles against the registered session providers."""

from collections.abc import Iterable
from uuid import UUID

from feedbax.contracts.execution_target import (
    LOCAL_PROFILE,
    LOCAL_RUN_KINDS,
    LOCAL_TARGET_ID,
    SESSION_RUN_KINDS,
    ExecutionTargets,
    ReadyTarget,
    SessionTargetProfile,
    TargetOption,
    TargetRunKind,
    UnavailableTarget,
)
from feedbax.orchestration.session import SessionError, SessionProvider, SessionRequest, session_capability_problem


class TargetRegistry:
    def __init__(
        self, profiles: Iterable[SessionTargetProfile] = (), providers: Iterable[SessionProvider] = (),
    ) -> None:
        self._profiles = self._index((profile.id, profile.model_copy(deep=True)) for profile in profiles)
        self._providers = self._index((provider.target, provider) for provider in providers)

    @staticmethod
    def _index[T](entries: Iterable[tuple[str, T]]) -> dict[str, T]:
        indexed = {}
        for key, entry in entries:
            if key in indexed:
                raise ValueError(f"Duplicate target registry key {key!r}")
            indexed[key] = entry
        return indexed

    def _unavailable(self, profile: SessionTargetProfile, kinds: Iterable[TargetRunKind]) -> str | None:
        provider = self._providers.get(profile.driver)
        if provider is None:
            return f"Session driver {profile.driver!r} is not registered."
        if problem := session_capability_problem(provider, profile.lifecycle):
            return problem
        if unsupported := set(kinds) - SESSION_RUN_KINDS:
            return f"The driver does not run these request kinds: {', '.join(sorted(unsupported))}."
        return None

    def catalog(
        self, *, kinds: Iterable[TargetRunKind] = (), session_submission: bool,
    ) -> ExecutionTargets:
        kinds = tuple(kinds)
        options = [TargetOption(
            profile=LOCAL_PROFILE, run_kinds=LOCAL_RUN_KINDS, availability=ReadyTarget(state="ready"),
        )]
        for profile in self._profiles.values():
            reason = self._unavailable(profile, kinds)
            if reason is None and not session_submission:
                reason = "Session submission is not configured."
            options.append(TargetOption(
                profile=profile.model_copy(deep=True), run_kinds=tuple(sorted(SESSION_RUN_KINDS)),
                availability=(
                    ReadyTarget(state="ready") if reason is None else UnavailableTarget(state="unavailable", reason=reason)
                ),
            ))
        return ExecutionTargets(default=LOCAL_TARGET_ID, targets=tuple(options))

    def require_session(
        self, target: str, *, kinds: Iterable[TargetRunKind] = (),
    ) -> SessionTargetProfile:
        if (profile := self._profiles.get(target)) is None:
            raise SessionError(f"Unknown execution target {target!r}.")
        if reason := self._unavailable(profile, kinds):
            raise SessionError(reason)
        return profile.model_copy(deep=True)

    def session_request(self, target: str, *, session_id: UUID, environment: str) -> SessionRequest:
        """Resolve the exact profile before requesting existing session authorization."""
        profile = self.require_session(target)
        return SessionRequest(
            id=session_id, target=profile.driver, profile=profile.id, resources=profile.resources,
            environment=environment, lifecycle=profile.lifecycle, maximum_microusd=profile.maximum_microusd,
        )
