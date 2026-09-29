"""Tool-use step for LLM rider agents.

Three tools exposed to the model via the provider's native tool-calling API.
All three read the processed trip Parquet (``data/processed/trips.parquet``)
through DuckDB -- no mocked or hardcoded distributions. Where a lane has too
little history, estimates fall back to the origin zone, then citywide.

Notes:
- Tools are bound to a specific rider id, so ``check_price_history`` can
  return "this rider's" history separately from the market's.
- ``check_transit_alternative`` derives transit minutes from the lane's real
  median trip miles (at a 12 mph transit speed) because the TLC data contains
  no transit feed; the $2.90 base fare is the MTA flat fare. Documented
  honestly: transit minutes are a data-derived heuristic, not a GTFS feed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import duckdb

from rider_sim.config import TRIPS_PATH

_TRANSIT_SPEED_MPH = 12.0
_TRANSIT_BASE_FARE = 2.90
_MIN_LANE_TRIPS = 20
_MIN_ZONE_TRIPS = 100


def _scalar(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any]) -> float | None:
    row = con.execute(sql, params).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _round2(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


@dataclass(frozen=True)
class PriceHistoryTool:
    """Historical fare percentiles for a lane, for this rider and the market."""

    rider_id: int
    trips_path: Path = TRIPS_PATH

    @property
    def name(self) -> str:
        return "check_price_history"

    @property
    def description(self) -> str:
        return (
            "Historical fares for a lane (pickup zone to dropoff zone). Returns the "
            "market's fare percentiles (p50/p90), the number of trips, and this "
            "rider's own median fare and its percentile rank within the market."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "origin_zone": {"type": "integer", "minimum": 1},
                "dest_zone": {"type": "integer", "minimum": 1},
            },
            "required": ["origin_zone", "dest_zone"],
        }

    def call(self, origin_zone: int, dest_zone: int) -> dict[str, Any]:
        con = duckdb.connect()
        try:
            lane_count = (
                _scalar(
                    con,
                    "SELECT count(*) FROM read_parquet(?) "
                    "WHERE pickup_zone = ? AND dropoff_zone = ?",
                    [str(self.trips_path), origin_zone, dest_zone],
                )
                or 0.0
            )
            if lane_count < _MIN_LANE_TRIPS:
                origin_count = (
                    _scalar(
                        con,
                        "SELECT count(*) FROM read_parquet(?) WHERE pickup_zone = ?",
                        [str(self.trips_path), origin_zone],
                    )
                    or 0.0
                )
                if origin_count >= _MIN_ZONE_TRIPS:
                    lane_filter = "pickup_zone = ?"
                    lane_params: list[Any] = [str(self.trips_path), origin_zone]
                    scope = "origin_zone"
                else:
                    lane_filter = "1 = 1"
                    lane_params = [str(self.trips_path)]
                    scope = "citywide"
            else:
                lane_filter = "pickup_zone = ? AND dropoff_zone = ?"
                lane_params = [str(self.trips_path), origin_zone, dest_zone]
                scope = "lane"

            market_p50 = _scalar(
                con,
                f"SELECT median(fare_paid) FROM read_parquet(?) WHERE {lane_filter}",
                lane_params,
            )
            market_p90 = _scalar(
                con,
                f"SELECT quantile_cont(fare_paid, 0.9) FROM read_parquet(?) WHERE {lane_filter}",
                lane_params,
            )
            rider_median = _scalar(
                con,
                f"SELECT median(fare_paid) FROM read_parquet(?) "
                f"WHERE rider_id = ? AND {lane_filter}",
                [str(self.trips_path), self.rider_id, *lane_params[1:]],
            )
            rider_percentile = _scalar(
                con,
                f"SELECT avg(CASE WHEN fare_paid <= ? THEN 1.0 ELSE 0.0 END) "
                f"FROM read_parquet(?) WHERE {lane_filter}",
                [
                    rider_median if rider_median is not None else 0.0,
                    *lane_params,
                ],
            )
        finally:
            con.close()
        return {
            "scope": scope,
            "n_trips": int(lane_count),
            "market_fare_p50": _round2(market_p50),
            "market_fare_p90": _round2(market_p90),
            "rider_median_fare": _round2(rider_median),
            "rider_fare_percentile": _round2(rider_percentile),
        }


@dataclass(frozen=True)
class EtaReliabilityTool:
    """Historical wait (request -> pickup) distribution for a zone-hour."""

    trips_path: Path = TRIPS_PATH

    @property
    def name(self) -> str:
        return "check_eta_reliability"

    @property
    def description(self) -> str:
        return (
            "Historical request-to-pickup waits (in minutes) for a pickup zone at a "
            "given hour of day. Returns the median and p90 wait and the number of "
            "trips, which the rider can use to judge whether a quoted ETA is credible."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "origin_zone": {"type": "integer", "minimum": 1},
                "hour": {"type": "integer", "minimum": 0, "maximum": 23},
            },
            "required": ["origin_zone", "hour"],
        }

    def call(self, origin_zone: int, hour: int) -> dict[str, Any]:
        con = duckdb.connect()
        try:
            n_trips = (
                _scalar(
                    con,
                    "SELECT count(*) FROM read_parquet(?) "
                    "WHERE pickup_zone = ? AND pickup_hour = ?",
                    [str(self.trips_path), origin_zone, hour],
                )
                or 0.0
            )
            if n_trips < _MIN_ZONE_TRIPS:
                zone_filter = "pickup_zone = ?"
                zone_params: list[Any] = [str(self.trips_path), origin_zone]
                scope = "origin_zone"
            else:
                zone_filter = "pickup_zone = ? AND pickup_hour = ?"
                zone_params = [str(self.trips_path), origin_zone, hour]
                scope = "zone_hour"
            p50 = _scalar(
                con,
                f"SELECT median(wait_secs) / 60.0 FROM read_parquet(?) WHERE {zone_filter}",
                zone_params,
            )
            p90 = _scalar(
                con,
                f"SELECT quantile_cont(wait_secs, 0.9) / 60.0 FROM read_parquet(?) "
                f"WHERE {zone_filter}",
                zone_params,
            )
        finally:
            con.close()
        return {
            "scope": scope,
            "n_trips": int(n_trips),
            "wait_p50_minutes": _round2(p50),
            "wait_p90_minutes": _round2(p90),
        }


@dataclass(frozen=True)
class TransitAlternativeTool:
    """Transit time/cost estimate derived from lane geometry in the trip data."""

    trips_path: Path = TRIPS_PATH

    @property
    def name(self) -> str:
        return "check_transit_alternative"

    @property
    def description(self) -> str:
        return (
            "Estimated transit alternative for a lane: travel minutes (derived from "
            "the lane's median trip miles at 12 mph transit speed) and the flat "
            "$2.90 base fare. Returns None minutes for lanes with no history."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "origin_zone": {"type": "integer", "minimum": 1},
                "dest_zone": {"type": "integer", "minimum": 1},
            },
            "required": ["origin_zone", "dest_zone"],
        }

    def call(self, origin_zone: int, dest_zone: int) -> dict[str, Any]:
        con = duckdb.connect()
        try:
            lane_miles = _scalar(
                con,
                "SELECT median(trip_miles) FROM read_parquet(?) "
                "WHERE pickup_zone = ? AND dropoff_zone = ?",
                [str(self.trips_path), origin_zone, dest_zone],
            )
            if lane_miles is None:
                lane_miles = _scalar(
                    con,
                    "SELECT median(trip_miles) FROM read_parquet(?) WHERE pickup_zone = ?",
                    [str(self.trips_path), origin_zone],
                )
            n_trips = (
                _scalar(
                    con,
                    "SELECT count(*) FROM read_parquet(?) "
                    "WHERE pickup_zone = ? AND dropoff_zone = ?",
                    [str(self.trips_path), origin_zone, dest_zone],
                )
                or 0.0
            )
        finally:
            con.close()
        minutes = None if lane_miles is None else round(lane_miles / _TRANSIT_SPEED_MPH * 60.0, 1)
        return {
            "transit_minutes": minutes,
            "transit_cost": _TRANSIT_BASE_FARE if minutes is not None else None,
            "n_trips": int(n_trips),
        }


class ToolRegistry:
    """Name -> tool, plus JSON specs for the provider tool-calling API."""

    def __init__(self, tools: list[Any]) -> None:
        self._tools: dict[str, Any] = {tool.name: tool for tool in tools}

    @property
    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return cast(dict[str, Any], tool.call(**arguments))
        except (TypeError, ValueError, KeyError) as exc:
            return {"error": f"invalid arguments for {name}: {exc}"}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def build_rider_tools(rider_id: int, trips_path: Path = TRIPS_PATH) -> ToolRegistry:
    """The standard tool suite for one rider agent."""
    return ToolRegistry(
        [
            PriceHistoryTool(rider_id=rider_id, trips_path=trips_path),
            EtaReliabilityTool(trips_path=trips_path),
            TransitAlternativeTool(trips_path=trips_path),
        ]
    )


def tool_result_message(name: str, result: dict[str, Any]) -> str:
    return json.dumps({"tool": name, "result": result}, sort_keys=True)
