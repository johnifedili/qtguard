from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class AuditView(BaseModel):
    missing_data: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class QTGuardOutput(BaseModel):
    risk_summary: str
    action_plan: List[str]
    patient_counseling: str
    audit_view: AuditView

    @classmethod
    def deferral(cls, missing: List[str]) -> "QTGuardOutput":
        return cls(
            risk_summary="Insufficient information to assess QT risk reliably.",
            action_plan=[
                "Request missing clinical inputs before providing a risk triage.",
                "Do not make medication changes based solely on this tool output.",
            ],
            patient_counseling=(
                "To provide safer guidance, we need a complete set of key measurements. "
                "Please ask your care team to confirm the missing items."
            ),
            audit_view=AuditView(
                missing_data=missing,
                notes=[
                    "Safe deferral triggered: required inputs not found in mini-chart.",
                    "This is a decision-support demo, not a medical device.",
                ],
            ),
        )



