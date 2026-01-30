# QTGuard Knowledge Base (Starter)

## Required inputs for QT risk triage
- QTc (ms)
- Potassium (K)
- Magnesium (Mg)
- Current medication list (with doses)
- Renal/hepatic function if available

## Common QT/TdP risk factors
- QTc >= 500 ms or large delta increase from baseline
- Hypokalemia / hypomagnesemia
- Multiple QT-prolonging meds
- Bradycardia
- Structural heart disease
- Recent dose increases / drug interactions that raise concentrations

## Output rules (demo-safe)
- If QTc/K/Mg missing → safe deferral + request missing inputs
- If evidence does not support a claim → do not assert it; ask a follow-up question
