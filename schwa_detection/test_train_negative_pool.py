import unittest

from train_negative_pool import eligible_window


class NegativeWindowTests(unittest.TestCase):
    def setUp(self):
        self.phone = {'phone': 'ah', 'start': .5, 'end': .56}

    def test_entire_window_must_avoid_schwa_and_unknown(self):
        for symbol in ('ax', 'unk'):
            other = {'phone': symbol, 'start': .30, 'end': .34}
            self.assertIsNone(eligible_window(self.phone, [self.phone, other]))

    def test_touching_boundary_is_allowed_but_overlap_is_not(self):
        touching = {'phone': 'ax', 'start': .78, 'end': .84}
        self.assertEqual(eligible_window(self.phone, [self.phone, touching]), (4480, 12480))
        touching['start'] = .779
        self.assertIsNone(eligible_window(self.phone, [self.phone, touching]))

    def test_short_phone_or_target_is_never_negative(self):
        for phone in ({'phone': 'ah', 'start': .5, 'end': .51},
                      {'phone': 'ax', 'start': .5, 'end': .56},
                      {'phone': 'unk', 'start': .5, 'end': .56}):
            self.assertIsNone(eligible_window(phone, [phone]))


if __name__ == '__main__':
    unittest.main()
