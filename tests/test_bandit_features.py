"""Запуск: python -m unittest discover -s tests -p 'test_bandit_features.py'."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.bandit_features import (
    FeatureArm, _prepare_search, build_feature, generate_arms,
    prepare_candidates, search_features,
)


class BanditFeaturesTest(unittest.TestCase):
    def test_library_includes_manual_features_as_candidates(self):
        names = [arm.name for arm in generate_arms()]
        self.assertEqual(len(names), 85)
        self.assertEqual(len(set(names)), 85)
        self.assertIn('history_length', names)
        self.assertIn('title_length', names)
        self.assertIn('user_previous_ctr', names)
        self.assertIn('user_category_previous_clicks', names)

    def test_column_arm_cannot_read_label(self):
        data = pd.DataFrame({'label': [0, 1], 'history_length': [2, 5]})
        result = build_feature(data, FeatureArm(column='history_length'))
        np.testing.assert_array_equal(result, [2, 5])
        with self.assertRaises(ValueError):
            build_feature(data, FeatureArm(column='label'))

    def test_history_matches_explicit_past_filter(self):
        # Непоследовательные строки, одинаковое время, граница окна и NaN-ключ.
        data = pd.DataFrame({
            'user_id': ['a', 'a', 'a', 'a', None, None],
            'timestamp': pd.to_datetime(['2020-01-02', '2020-01-01',
                                         '2020-01-02', '2020-01-03',
                                         '2020-01-01', '2020-01-02']),
            'label': [1, 1, 0, 0, 1, 0],
        }, index=[8, 4, 8, 2, 9, 0])
        for window in ('all', '1d'):
            for statistic in ('impressions', 'clicks', 'ctr'):
                arm = FeatureArm('user', statistic, window)
                expected = []
                for _, row in data.iterrows():
                    same_user = (data.user_id.isna() if pd.isna(row.user_id)
                                 else data.user_id.eq(row.user_id))
                    mask = same_user & data.timestamp.lt(row.timestamp)
                    if window != 'all':
                        mask &= data.timestamp.ge(
                            row.timestamp - pd.Timedelta(window.replace('d', 'D')),
                        )
                    past = data[mask]
                    count, clicks = len(past), past.label.sum()
                    expected.append({'impressions': count, 'clicks': clicks,
                                     'ctr': clicks / count if count else 0}[statistic])
                result = build_feature(data, arm)
                np.testing.assert_allclose(result, expected)
                self.assertTrue(result.index.equals(data.index))

        # Изменение текущих/будущих меток не меняет более ранние признаки.
        changed = data.copy()
        mask = changed.timestamp.ge(pd.Timestamp('2020-01-02'))
        changed.loc[mask, 'label'] = 1 - changed.loc[mask, 'label']
        arm = FeatureArm('user', 'clicks', 'all')
        early = data.timestamp.le(pd.Timestamp('2020-01-02'))
        np.testing.assert_allclose(build_feature(data, arm)[early],
                                   build_feature(changed, arm)[early])

    def make_data(self):
        """Вернуть небольшой поток выдач с обоими классами."""
        return pd.DataFrame({
            'impression_id': np.repeat(np.arange(20), 2),
            'timestamp': np.repeat(pd.date_range('2020-01-01', periods=20, freq='h'), 2),
            'item_id': ['a', 'b'] * 20,
            'category': ['sport', 'news'] * 20,
            'subcategory': ['football', 'world'] * 20,
            'label': [1, 0] * 20,
            'history_length': np.repeat(np.arange(20), 2),
            'title_length': np.repeat(np.arange(20) ** 2, 2),
            'history_contains_item': np.zeros(40),
            'item_previous_ctr': np.linspace(0, 1, 40),
        })

    def test_control_is_hidden_from_search_and_filtering(self):
        data = self.make_data()
        # Граница проходит по timestamp: соседние выдачи в один момент
        # не должны оказаться по разные стороны.
        data.loc[data.impression_id == 15, 'timestamp'] = data.loc[
            data.impression_id == 16, 'timestamp'
        ].iloc[0]
        full, early, folds, control = _prepare_search(
            data, ['category', 'subcategory'], .2,
        )
        train_indices, valid_indices = control
        self.assertLess(full.iloc[train_indices].timestamp.max(),
                        full.iloc[valid_indices].timestamp.min())
        self.assertTrue(set(early.impression_id).isdisjoint(
            full.iloc[valid_indices].impression_id))
        for train, valid in folds:
            self.assertLess(early.iloc[train].timestamp.max(),
                            early.iloc[valid].timestamp.min())

        changed = data.copy()
        changed.loc[valid_indices, 'label'] = 1 - changed.loc[valid_indices, 'label']
        changed.loc[valid_indices, 'history_length'] = 999999
        _, early_changed, _, _ = _prepare_search(
            changed, ['category', 'subcategory'], .2,
        )
        pd.testing.assert_frame_equal(early, early_changed)

    def test_constants_and_duplicates_are_removed(self):
        data = self.make_data()
        data['title_length'] = data['history_length']
        arms = [
            FeatureArm(column='history_length'),
            FeatureArm(column='title_length'),
            FeatureArm(column='history_contains_item'),
        ]
        _, kept, discarded = prepare_candidates(data, arms, ['category', 'subcategory'])
        self.assertEqual(kept, [arms[0]])
        self.assertEqual(discarded.reason.tolist(),
                         ['duplicate:history_length', 'constant'])

    def test_paired_rewards_and_cheaper_screening(self):
        data = self.make_data()
        arms = [
            FeatureArm('item', 'clicks', '6h'),
            FeatureArm('item', 'clicks', '1d'),
            FeatureArm(column='history_length'),
        ]
        observed = []

        def fake_score(table, features, fold, params):
            observed.append((len(table), features.copy(), params.copy()))
            if len(table) == len(data):
                return .46 if 'history_length' in features else .4
            if params['iterations'] == 50:
                return .2 if len(features) > 2 else .1
            return .45 if 'history_length' in features else (
                .42 if len(features) > 2 else .4)

        with patch('src.bandit_features.evaluate_features', side_effect=fake_score), \
             patch('src.bandit_features.train_model') as train:
            result = search_features(data, arms=arms, budget=30, max_features=1)

        self.assertEqual(result.selected_arms, [arms[2]])
        self.assertEqual(result.feature_columns,
                         ['category', 'subcategory', 'history_length'])
        self.assertNotIn('item_previous_ctr', result.feature_columns)
        self.assertEqual(result.evaluations, 30)
        self.assertEqual(result.trees_built, 8000)
        self.assertAlmostEqual(result.cv_score, .45)
        self.assertAlmostEqual(result.control_score, .46)
        self.assertEqual(set(result.history[result.history.stage == 'screen'].iterations), {50})
        self.assertEqual(set(result.history[result.history.stage == 'full'].iterations), {300})
        np.testing.assert_allclose(
            result.history.reward, result.history.score - result.history.baseline_score,
        )
        # Обе модели каждого парного сравнения имеют одинаковые seed/фолды.
        full = result.history[result.history.stage == 'full']
        self.assertEqual(set(full.seed), {42, 0})
        self.assertEqual(set(full.fold), {1, 2, 3})
        # Только baseline и один победитель видят поздний контроль.
        self.assertEqual(sum(n == len(data) for n, _, _ in observed), 2)
        self.assertEqual(len(train.call_args.args[0]), len(data))
        self.assertEqual(train.call_args.args[1], result.feature_columns)

    def test_best_control_checkpoint_is_restored(self):
        data = self.make_data()
        arms = [
            FeatureArm(column='history_length'),
            FeatureArm(column='title_length'),
        ]

        def fake_score(table, features, fold, params):
            if len(table) == len(data):
                return {2: .4, 3: .46, 4: .43}[len(features)]
            # На поисковых данных каждый шаг улучшает CV; контроль второго
            # шага хуже. Итог обязан вернуться к одному добавлению.
            gain = .05 if 'history_length' in features else 0
            gain += .02 if 'title_length' in features else 0
            return .4 + gain

        with patch('src.bandit_features.evaluate_features', side_effect=fake_score), \
             patch('src.bandit_features.train_model') as train:
            result = search_features(data, arms=arms, budget=100, max_features=2)

        self.assertEqual(result.selected_arms, [arms[0]])
        self.assertEqual(result.checkpoints.step.tolist(), [0, 1, 2])
        self.assertEqual(result.checkpoints.selected.tolist(), [False, True, False])
        self.assertAlmostEqual(result.control_score, .46)
        self.assertEqual(train.call_args.args[1],
                         ['category', 'subcategory', 'history_length'])

    def test_no_gain_keeps_only_categories(self):
        data = self.make_data()
        with patch('src.bandit_features.evaluate_features', return_value=.4), \
             patch('src.bandit_features.train_model'):
            result = search_features(
                data, arms=[FeatureArm(column='history_length')], budget=16,
            )
        self.assertEqual(result.selected_arms, [])
        self.assertEqual(result.feature_columns, ['category', 'subcategory'])
        self.assertEqual(result.checkpoints.step.tolist(), [0])
        self.assertEqual(result.evaluations, 15)

    def test_control_patience_stops_after_two_bad_steps(self):
        data = self.make_data()
        data['abstract_length'] = np.repeat(np.arange(20) ** 3, 2)
        arms = [FeatureArm(column=c) for c in
                ('history_length', 'title_length', 'abstract_length')]

        def fake_score(table, features, fold, params):
            if len(table) == len(data):
                return .4 if len(features) == 2 else .39
            weights = {'history_length': .03, 'title_length': .02, 'abstract_length': .01}
            return .4 + sum(weights.get(c, 0) for c in features)

        with patch('src.bandit_features.evaluate_features', side_effect=fake_score), \
             patch('src.bandit_features.train_model'):
            result = search_features(
                data, arms=arms, budget=200, max_features=3, patience=2,
            )
        self.assertEqual(result.checkpoints.step.tolist(), [0, 1, 2])
        self.assertEqual(result.selected_arms, [])
        self.assertEqual(result.feature_columns, ['category', 'subcategory'])

    def test_budget_checked_before_round(self):
        data = self.make_data()
        with patch('src.bandit_features.evaluate_features', return_value=.4), \
             patch('src.bandit_features.train_model'):
            result = search_features(
                data, arms=[FeatureArm(column='history_length')], budget=7,
            )
        self.assertEqual(result.evaluations, 7)
        self.assertEqual(result.selected_arms, [])
        self.assertEqual(result.trees_built, 2100)


if __name__ == '__main__':
    unittest.main()
