from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auditor_agent import Action, AuditAgent, LlmFinding, VectorPolicyIndex, load_cases, load_rules


class FakeLlm:
    def analyze_many(self, items):
        return {case.id_caso: self.analyze(case, policies) for case, policies in items}

    def analyze(self, case, policies):
        if case.id_caso == 1:
            action, risk = Action.APPROVE, "bajo"
        elif case.id_caso == 2:
            action, risk = Action.ESCALATE, "medio"
        elif case.id_caso == 3:
            action, risk = Action.APPROVE, "alto"
        else:
            action, risk = Action.APPROVE, "critico"
        return LlmFinding(
            accion_detectada=action,
            contradiccion_normativa=case.id_caso in {3, 4},
            nivel_riesgo=risk,
            razon=f"Analisis estructurado del caso {case.id_caso}.",
        )


class AuditorAgentTest(unittest.TestCase):
    def test_expected_decisions_for_challenge_cases(self):
        rules = load_rules(ROOT / "reglas.json")
        cases = load_cases(ROOT / "data" / "casos_agent_b.json")
        index = VectorPolicyIndex(ROOT / "data" / "politicas")
        results = {result.id_caso: result for result in AuditAgent(rules, index, FakeLlm()).audit(cases)}

        self.assertEqual(results[1].estado_transaccion, "CONFORME - APROBADA")
        self.assertEqual(results[2].estado_transaccion, "CONFORME - ESCALADA A ANALISTA")
        self.assertEqual(results[3].estado_transaccion, "NO CONFORME - REQUIERE REVISION")
        self.assertEqual(results[4].estado_transaccion, "NO CONFORME - BLOQUEO CRITICO")
        self.assertTrue(results[4].fuentes_rag)


if __name__ == "__main__":
    unittest.main()
