# tests/test_planner/test_alerts.py
"""Tests for 15 alert types across 4 severity levels."""
from __future__ import annotations

import pytest

from stellaris_advisor.config import PlannerConfig
from stellaris_advisor.parser.models import EmpireState, GalacticState


@pytest.mark.unit
class TestGenerateAlerts:
    def test_returns_tuple_of_alerts(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        from stellaris_advisor.planner.alerts import generate_alerts
        from stellaris_advisor.planner.models import Alert

        config = PlannerConfig()
        alerts = generate_alerts(planner_empire, planner_galactic, config)
        assert isinstance(alerts, tuple)
        assert all(isinstance(a, Alert) for a in alerts)

    def test_hostile_neighbor_alert(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        """Hostile empire has 15000 fleet vs player's 8000 -> ratio 1.875 > 1.5 threshold."""
        from stellaris_advisor.planner.alerts import generate_alerts
        from stellaris_advisor.planner.models import AlertSeverity

        config = PlannerConfig(threat_power_ratio=1.5)
        alerts = generate_alerts(planner_empire, planner_galactic, config)
        high_alerts = [a for a in alerts if a.severity == AlertSeverity.HIGH]
        threat_alerts = [
            a
            for a in high_alerts
            if "threat" in a.category or "hostile" in a.message.lower()
        ]
        assert len(threat_alerts) >= 1

    def test_no_ascension_path_alert(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        """Year 2275, no ascension path -> should not fire yet (deadline 2300)."""
        from stellaris_advisor.planner.alerts import generate_alerts
        from stellaris_advisor.planner.models import AlertSeverity

        config = PlannerConfig(ascension_path_deadline_year=2300)
        alerts = generate_alerts(planner_empire, planner_galactic, config)
        path_alerts = [
            a
            for a in alerts
            if "ascension" in a.message.lower() and a.severity == AlertSeverity.HIGH
        ]
        assert len(path_alerts) == 0  # year 2275 < 2280 threshold

    def test_naval_underutilization_alert(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        """Naval 35/80 = 0.4375 < 0.5 -> low severity alert."""
        from stellaris_advisor.planner.alerts import generate_alerts
        from stellaris_advisor.planner.models import AlertSeverity

        config = PlannerConfig()
        alerts = generate_alerts(planner_empire, planner_galactic, config)
        naval_alerts = [
            a
            for a in alerts
            if a.severity == AlertSeverity.LOW and "naval" in a.message.lower()
        ]
        assert len(naval_alerts) >= 1

    def test_alert_count_within_bounds(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        from stellaris_advisor.planner.alerts import generate_alerts

        config = PlannerConfig()
        alerts = generate_alerts(planner_empire, planner_galactic, config)
        assert len(alerts) <= 15
