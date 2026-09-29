"""Append-only episode memory for rider agents.

Each rider has a ``RiderMemory``: an append-only log of episodes (past offers
and the chosen action, plus realized outcomes when they arrive). ``recall``
returns a compact, deterministic, JSON-serializable summary of the k most
relevant past episodes plus rolling aggregates. This summary is what gets
injected into the LLM prompt; nothing else is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rider_sim.agent.schemas import Action, RideOffer

_BAD_WAIT_SLACK_MINUTES = 10.0
_HIGH_SURGE_THRESHOLD = 1.5


@dataclass
class Episode:
    """One (offer, decision, outcome) triple in a rider's history."""

    offer: RideOffer
    action: Action
    realized_fare: float | None = None
    realized_wait: float | None = None
    satisfaction_delta: float = 0.0


def _surge_band(surge: float) -> float:
    return min(round(surge * 2.0) / 2.0, 5.0)


@dataclass
class RiderMemory:
    """Append-only episode log for a single pseudo-rider."""

    rider_id: int
    episodes: list[Episode] = field(default_factory=list)

    def append(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def append_decision(self, offer: RideOffer, action: Action) -> Episode:
        episode = Episode(offer=offer, action=action)
        self.episodes.append(episode)
        return episode

    def record_realization(
        self,
        offer_id: str,
        realized_fare: float | None = None,
        realized_wait: float | None = None,
        satisfaction_delta: float = 0.0,
    ) -> None:
        """Fill in the realized outcome for the most recent episode of an offer.

        The episode list stays append-only: this only mutates the outcome
        fields of an existing episode, never adds, removes, or reorders.
        """
        for episode in reversed(self.episodes):
            if episode.offer.offer_id == offer_id:
                episode.realized_fare = realized_fare
                episode.realized_wait = realized_wait
                episode.satisfaction_delta = satisfaction_delta
                return

    def _similarity(self, episode: Episode, offer: RideOffer) -> float:
        purpose = 0.4 if episode.offer.trip_purpose == offer.trip_purpose else 0.0
        zone = 0.3 if episode.offer.origin_zone == offer.origin_zone else 0.0
        surge = (
            0.3
            if _surge_band(episode.offer.surge_multiplier) == _surge_band(offer.surge_multiplier)
            else 0.0
        )
        return purpose + zone + surge

    def relevant_episodes(self, offer: RideOffer, k: int) -> list[Episode]:
        """k most relevant episodes: by similarity, ties broken by recency."""
        ranked = sorted(
            enumerate(self.episodes),
            key=lambda pair: (self._similarity(pair[1], offer), pair[0]),
            reverse=True,
        )
        return [episode for _, episode in ranked[: max(k, 0)]]

    def aggregates(self) -> dict[str, float | int | None]:
        """Rolling aggregates over the whole episode log."""
        realized_fares = [e.realized_fare for e in self.episodes if e.realized_fare is not None]
        high_surge = [e for e in self.episodes if e.offer.surge_multiplier > _HIGH_SURGE_THRESHOLD]
        bad_waits = [
            e
            for e in self.episodes
            if e.realized_wait is not None
            and e.realized_wait > e.offer.eta_minutes + _BAD_WAIT_SLACK_MINUTES
        ]
        return {
            "episodes_total": len(self.episodes),
            "mean_fare_paid": round(sum(realized_fares) / len(realized_fares), 2)
            if realized_fares
            else None,
            "accept_rate_high_surge": round(
                sum(1 for e in high_surge if e.action == "accept") / len(high_surge), 3
            )
            if high_surge
            else None,
            "bad_wait_experiences": len(bad_waits),
        }

    def recall(self, offer: RideOffer, k: int = 5) -> dict[str, object]:
        """Compact deterministic memory snapshot for prompt injection."""
        episodes_payload: list[dict[str, object]] = []
        for episode in self.relevant_episodes(offer, k):
            episodes_payload.append(
                {
                    "purpose": episode.offer.trip_purpose,
                    "origin_zone": episode.offer.origin_zone,
                    "dest_zone": episode.offer.dest_zone,
                    "quoted_fare": round(episode.offer.quoted_fare, 2),
                    "surge": round(episode.offer.surge_multiplier, 2),
                    "eta_minutes": round(episode.offer.eta_minutes, 1),
                    "action": episode.action,
                    "realized_fare": round(episode.realized_fare, 2)
                    if episode.realized_fare is not None
                    else None,
                    "realized_wait": round(episode.realized_wait, 1)
                    if episode.realized_wait is not None
                    else None,
                    "satisfaction_delta": round(episode.satisfaction_delta, 2),
                }
            )
        return {
            "rider_id": self.rider_id,
            "relevant_episodes": episodes_payload,
            "aggregates": self.aggregates(),
        }

    def __len__(self) -> int:
        return len(self.episodes)
