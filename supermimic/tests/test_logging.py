import contextlib
import io
import json
import unittest

from supermimic.core.logging import progress


class LoggingTests(unittest.TestCase):
    def test_terminal_hides_diagnostics_without_mutating_tensorboard_metrics(self):
        metrics = dict(update=50, first_interval_includes_compile=False, eta_seconds=120,
                       elapsed_seconds=60, updates_per_second=3, transitions_per_second=24000,
                       transitions=100000, critic_loss=10, critic_loss_peak=500,
                       wm_loss=.1, discriminator_loss=.3, critic_target_abs_max=1000,
                       critic_value_abs_max=5, imagined_reward_abs_max=900, imagined_return=.3,
                       actor_gradient_norm=2, actor_gradient_norm_peak=30, recovery_mode_fraction=.4,
                       guard_events_interval=1, guard_envs_interval=2, guard_events_total=3,
                       interval_updates=50)
        for mode in ('pretty', 'json'):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                progress(metrics, 1000, mode)
            text = output.getvalue()
            if mode == 'json':
                displayed=json.loads(text)
                self.assertNotIn('critic_target_abs_max',displayed)
                self.assertNotIn('guard_events_total',displayed)
                self.assertEqual(displayed['critic_loss'],metrics['critic_loss'])
            else:
                self.assertIn('interval peak 500', text)
                self.assertIn('ETA 00:02:00', text)
                self.assertNotIn('CHECK', text)
                self.assertNotIn('Guards', text)
                self.assertNotIn('max |target|', text)
                self.assertNotIn('Stats:', text)
            self.assertEqual(metrics['critic_target_abs_max'],1000)
            self.assertEqual(metrics['guard_events_total'],3)
