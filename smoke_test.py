from qtguard_core.guardrails import build_safe_output

if __name__ == "__main__":
    print(build_safe_output("QTc=520 ms; K=3.1; Mg=1.6; Meds: ondansetron").model_dump())
    print(build_safe_output("Meds: ondansetron only").model_dump())
