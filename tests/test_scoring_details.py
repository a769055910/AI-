import unittest

import app as web


class ScoringDetailsTests(unittest.TestCase):
    def assert_trace_matches_result(self, result):
        trace = result['scoring']
        self.assertAlmostEqual(sum(item['weight'] for item in trace['components']), 1.0)
        weighted_score = sum(item['score'] * item['weight'] for item in trace['components'])
        self.assertAlmostEqual(trace['base_score'], weighted_score, delta=0.00005)
        self.assertEqual(trace['base_score'], result['final_score'])

    def test_missing_ai_models_are_excluded_and_available_weights_normalized(self):
        result = web.compute_combined_verdict({'score': 0.2}, {
            'probe_dinov2': {'available': True, 'ai_score': 0.8},
            'univfd': {'available': False, 'ai_score': 0.99},
        })
        self.assert_trace_matches_result(result)
        self.assertEqual([item['name'] for item in result['scoring']['components']], ['NPR', 'PROBE-DINOv2'])
        self.assertAlmostEqual(result['scoring']['components'][0]['weight'], 0.4)

    def test_removed_models_in_historical_payload_do_not_affect_scoring(self):
        current = {'probe_dinov2': {'available': True, 'ai_score': 0.8}}
        historical = {**current,
                      'aide': {'available': True, 'ai_score': 0.0},
                      'dear_r': {'available': True, 'ai_score': 0.0}}
        clean = web.compute_combined_verdict({'score': 0.6}, current)
        self.assertEqual(clean, web.compute_combined_verdict({'score': 0.6}, historical))
        self.assertAlmostEqual(clean['final_score'], 0.72)
        self.assertNotIn('aide_ai_score', clean)
        self.assertNotIn('dear_r_ai_score', clean)

    def test_deepfake_scoring_and_threshold_boundary(self):
        result = web.compute_deepfake_combined_verdict(
            {'deepfake_score': 0.6, 'face_detected': True, 'verdict': 'real'},
            {'deepfake_score': 0.6},
        )
        self.assert_trace_matches_result(result)
        self.assertEqual(result['scoring']['operator'], '>')
        self.assertNotIn('疑似', result['final_verdict'])
        no_face = web.compute_deepfake_combined_verdict({'face_detected': False}, None)
        self.assertEqual(no_face['scoring']['components'], [])
        self.assertIsNone(no_face['scoring']['base_score'])

    def test_tamper_auxiliary_scores_do_not_change_primary_score(self):
        result = web.compute_tamper_combined_verdict({
            'trufor': {'available': True, 'score': 0.6, 'reliability': 0.8},
            'copy_move': {'score': 0.99}, 'block_noise': {'score': 0.99},
        })
        self.assert_trace_matches_result(result)
        self.assertEqual(len(result['scoring']['components']), 1)
        self.assertEqual(result['final_score'], 0.6)
        self.assertEqual(result['scoring']['operator'], '>=')
        self.assertEqual(result['final_verdict'], '疑似图像篡改')
        unavailable = web.compute_tamper_combined_verdict({'trufor': {'available': False}})
        self.assertIsNone(unavailable['scoring']['base_score'])

    def test_source_adjustment_keeps_original_calculation(self):
        original = web.compute_combined_verdict({'score': 0.2}, {})
        adjusted = web.apply_c2pa_ai_evidence(original, {'c2pa_verification': {'present': True}})
        self.assertEqual(adjusted['scoring']['base_score'], 0.2)
        self.assertEqual(adjusted['scoring']['adjustment']['score'], adjusted['final_score'])
        self.assertEqual(adjusted['final_score'], 0.99)
        self.assertNotIn('adjustment', original['scoring'])


if __name__ == '__main__':
    unittest.main()
