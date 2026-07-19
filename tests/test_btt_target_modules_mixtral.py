import os, sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from btt_layer import get_blocktt_target_module_names


class TestMixtralTargetModules(unittest.TestCase):
    def test_mixtral_all_covers_attention_and_experts(self):
        names = get_blocktt_target_module_names("mixtral_all")
        # Mixtral attention leaf names
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            self.assertIn(n, names)
        # Mixtral MoE expert leaf names (w1=gate, w2=down, w3=up)
        for n in ("w1", "w2", "w3"):
            self.assertIn(n, names)

    def test_mixtral_all_excludes_router_and_llama_mlp_names(self):
        names = get_blocktt_target_module_names("mixtral_all")
        # router must never be adapted
        self.assertNotIn("gate", names)
        # Llama-style MLP names must not leak in (Mixtral has none)
        for n in ("gate_proj", "up_proj", "down_proj"):
            self.assertNotIn(n, names)

    def test_existing_types_unchanged(self):
        self.assertEqual(
            get_blocktt_target_module_names("all"),
            ("gate_proj", "up_proj", "down_proj",
             "q_proj", "k_proj", "v_proj", "o_proj"),
        )

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            get_blocktt_target_module_names("nonsense")


if __name__ == "__main__":
    unittest.main()
