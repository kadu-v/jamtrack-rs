import unittest

from scripts.gen_charts import (
    collect_mot_results,
    collect_performance_results,
    load_tracker_files,
    pareto_frontier,
    performance_label_levels,
    selected_mot_labels,
    variant_group,
)


def mot_result(label, implementation, hota, idsw, mota, idf1):
    return {
        "label": label,
        "chart_label": label,
        "implementation": implementation,
        "variant": "default",
        "metrics": {
            "hota": hota,
            "idsw": idsw,
            "mota": mota,
            "idf1": idf1,
        },
    }


class ParetoFrontierTests(unittest.TestCase):
    def test_mixed_maximize_and_minimize_frontier(self):
        results = [
            mot_result("quality", "rust", 10, 5, 0, 0),
            mot_result("stability", "rust", 9, 4, 0, 0),
            mot_result("dominated", "rust", 8, 6, 0, 0),
        ]

        frontier = pareto_frontier(
            results, "hota", "idsw", maximize_x=True, maximize_y=False
        )

        self.assertEqual({result["label"] for result in frontier}, {"quality", "stability"})

    def test_maximize_both_frontier(self):
        results = [
            mot_result("mota", "rust", 0, 0, 10, 9),
            mot_result("idf1", "rust", 0, 0, 9, 10),
            mot_result("dominated", "rust", 0, 0, 8, 8),
        ]

        frontier = pareto_frontier(
            results, "mota", "idf1", maximize_x=True, maximize_y=True
        )

        self.assertEqual({result["label"] for result in frontier}, {"mota", "idf1"})

    def test_label_selection_keeps_rust_and_pareto_python(self):
        results = [
            mot_result("rust", "rust", 10, 6, 10, 10),
            mot_result("python-frontier", "python", 11, 5, 9, 9),
            mot_result("python-dominated", "python", 9, 7, 8, 8),
        ]

        self.assertEqual(
            selected_mot_labels(results),
            {"rust", "python-frontier"},
        )


class ChartDataTests(unittest.TestCase):
    def test_performance_labels_use_separate_levels_for_nearby_values(self):
        levels = performance_label_levels([77.9, 80.9, 84.1])

        self.assertEqual(len(set(levels)), 3)

    def test_variant_groups(self):
        self.assertEqual(variant_group("default"), "Default")
        self.assertEqual(variant_group("tuned"), "Tuned")
        self.assertEqual(variant_group("plusplus"), "Enhanced")
        self.assertEqual(variant_group("plusplus_ecc"), "ECC")

    def test_repository_benchmark_data_is_complete(self):
        trackers = load_tracker_files()
        performance_results, _ = collect_performance_results(trackers)
        mot_results, _ = collect_mot_results(trackers)

        self.assertEqual(len(trackers), 5)
        self.assertEqual(len(performance_results), 7)
        self.assertEqual(len(mot_results), 18)
        self.assertTrue(all(result["chart_label"] for result in mot_results))


if __name__ == "__main__":
    unittest.main()
