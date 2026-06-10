# 文件名: test_spec_registry.py - 推测解码注册表
"""Unit tests for the speculative algorithm plugin registry."""

import unittest
from unittest.mock import MagicMock

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import (
    _REGISTRY,
    _RESERVED_NAMES,
    CustomSpecAlgo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# _RegistryIsolated类
class _RegistryIsolated(CustomTestCase):
    """Snapshot and restore the global registry so tests don't leak."""

    def setUp(self):
        self._snapshot = _REGISTRY.copy()
        _REGISTRY.clear()

    # _RegistryIsolated类的测试清理
    def tearDown(self):
        _REGISTRY.clear()
        _REGISTRY.update(self._snapshot)


# TestFromString类
class TestFromString(_RegistryIsolated):

    # TestFromString类的测试noneinputreturnsnonemember
    def test_none_input_returns_none_member(self):
        self.assertIs(SpeculativeAlgorithm.from_string(None), SpeculativeAlgorithm.NONE)  # 断言是同一对象

    # TestFromString类的测试builtinnamereturnsenum
    def test_builtin_name_returns_enum(self):
        self.assertIs(  # 断言是同一对象
            SpeculativeAlgorithm.from_string("EAGLE"), SpeculativeAlgorithm.EAGLE
        )
        self.assertIs(  # 断言是同一对象
            SpeculativeAlgorithm.from_string("NGRAM"), SpeculativeAlgorithm.NGRAM
        )

    # TestFromString类的测试builtinnameiscaseinsensitive
    def test_builtin_name_is_case_insensitive(self):
        self.assertIs(  # 断言是同一对象
            SpeculativeAlgorithm.from_string("eagle"), SpeculativeAlgorithm.EAGLE
        )

    # TestFromString类的测试unknownnameraises
    def test_unknown_name_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown speculative algorithm"):
            SpeculativeAlgorithm.from_string("NOT_REGISTERED")

    # TestFromString类的测试registeredpluginreturnscustomspec
    def test_registered_plugin_returns_custom_spec(self):
        @SpeculativeAlgorithm.register("MY_FOO")

        # 内部方法_factory
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string("MY_FOO")
        self.assertIsInstance(algo, CustomSpecAlgo)
        self.assertEqual(algo.name, "MY_FOO")  # 断言相等

    # TestFromString类的测试registeredpluginlookupiscaseinsensitive
    def test_registered_plugin_lookup_is_case_insensitive(self):
        @SpeculativeAlgorithm.register("MY_FOO")

        # 内部方法_factory
        def _factory(server_args):
            return MagicMock

        self.assertIs(  # 断言是同一对象
            SpeculativeAlgorithm.from_string("my_foo"),
            SpeculativeAlgorithm.from_string("MY_FOO"),
        )


# TestRegister类
class TestRegister(_RegistryIsolated):

    # TestRegister类的测试registerreturnsfactoryunchanged
    def test_register_returns_factory_unchanged(self):

        # 内部方法_factory
        def _factory(server_args):
            return MagicMock

        decorated = SpeculativeAlgorithm.register("MY_FOO")(_factory)
        self.assertIs(decorated, _factory)  # 断言是同一对象

    # TestRegister类的测试twodistinctregistrationsareindependent
    def test_two_distinct_registrations_are_independent(self):
        @SpeculativeAlgorithm.register("FOO")

        # 内部方法_foo_factory
        def _foo_factory(server_args):
            return MagicMock

        @SpeculativeAlgorithm.register("BAR")

        # 内部方法_bar_factory
        def _bar_factory(server_args):
            return MagicMock

        foo = SpeculativeAlgorithm.from_string("FOO")
        bar = SpeculativeAlgorithm.from_string("BAR")
        self.assertIsNot(foo, bar)  # 断言不是同一对象
        self.assertNotEqual(foo, bar)  # 断言不相等
        self.assertEqual(foo.name, "FOO")  # 断言相等
        self.assertEqual(bar.name, "BAR")  # 断言相等

    # TestRegister类的测试duplicatenameraises
    def test_duplicate_name_raises(self):
        @SpeculativeAlgorithm.register("MY_FOO")

        # 内部方法_factory
        def _factory(server_args):
            return MagicMock

        with self.assertRaisesRegex(ValueError, "already registered"):

            @SpeculativeAlgorithm.register("MY_FOO")

            # 内部方法_factory2
            def _factory2(server_args):
                return MagicMock

    # TestRegister类的测试reservednameraises
    def test_reserved_name_raises(self):
        for reserved in _RESERVED_NAMES:
            with self.assertRaisesRegex(ValueError, "reserved"):
                SpeculativeAlgorithm.register(reserved)

    # TestRegister类的测试registeriscaseinsensitiveoncollision
    def test_register_is_case_insensitive_on_collision(self):
        @SpeculativeAlgorithm.register("MY_FOO")

        # 内部方法_factory
        def _factory(server_args):
            return MagicMock

        with self.assertRaisesRegex(ValueError, "already registered"):

            @SpeculativeAlgorithm.register("my_foo")

            # 内部方法_factory2
            def _factory2(server_args):
                return MagicMock


# TestCustomSpecAlgoInterface类
class TestCustomSpecAlgoInterface(_RegistryIsolated):
    """CustomSpecAlgo must duck-type SpeculativeAlgorithm enum values."""

    def setUp(self):
        super().setUp()

        @SpeculativeAlgorithm.register("MY_FOO", supports_overlap=False)

        # 内部方法_factory
        def _factory(server_args):
            return MagicMock

        self.algo = SpeculativeAlgorithm.from_string("MY_FOO")

    # TestCustomSpecAlgoInterface类的测试ispredicatesallfalseexceptspeculative
    def test_is_predicates_all_false_except_speculative(self):
        self.assertFalse(self.algo.is_none())  # 断言为假
        self.assertFalse(self.algo.is_eagle())  # 断言为假
        self.assertFalse(self.algo.is_eagle3())  # 断言为假
        self.assertFalse(self.algo.is_dflash())  # 断言为假
        self.assertFalse(self.algo.is_standalone())  # 断言为假
        self.assertFalse(self.algo.is_ngram())  # 断言为假
        self.assertTrue(self.algo.is_speculative())  # 断言为真

    # TestCustomSpecAlgoInterface类的测试supportsspecv2followssupportsoverlap
    def test_supports_spec_v2_follows_supports_overlap(self):
        # Plugin registered with supports_overlap=False -> not spec_v2.
        self.assertFalse(self.algo.supports_spec_v2())  # 断言为假

        @SpeculativeAlgorithm.register("MY_V2", supports_overlap=True)

        # 内部方法_factory
        def _factory(server_args):
            return MagicMock

        v2 = SpeculativeAlgorithm.from_string("MY_V2")
        self.assertTrue(v2.supports_spec_v2())  # 断言为真

    # TestCustomSpecAlgoInterface类的测试createworkercallsfactory
    def test_create_worker_calls_factory(self):
        server_args = MagicMock()
        server_args.disable_overlap_schedule = True
        worker_cls = self.algo.create_worker(server_args)
        self.assertIs(worker_cls, MagicMock)  # 断言是同一对象

    # TestCustomSpecAlgoInterface类的测试createworkerraisesonoverlapmismatch
    def test_create_worker_raises_on_overlap_mismatch(self):
        server_args = MagicMock()
        server_args.disable_overlap_schedule = False
        with self.assertRaisesRegex(ValueError, "does not support overlap"):
            self.algo.create_worker(server_args)


# TestValidatorHook类
class TestValidatorHook(_RegistryIsolated):

    # TestValidatorHook类的测试validatorinvocationiscallerdriven
    def test_validator_invocation_is_caller_driven(self):
        validator = MagicMock()

        @SpeculativeAlgorithm.register("MY_FOO", validate_server_args=validator)

        # 内部方法_factory
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string("MY_FOO")
        self.assertIs(algo.validate_server_args, validator)  # 断言是同一对象
        # Callers (e.g. ServerArgs.__post_init__) must invoke the hook themselves;
        # CustomSpecAlgo does not call it from create_worker.
        validator.assert_not_called()


# TestSubclassOverride类
class TestSubclassOverride(_RegistryIsolated):
    """Plugins can subclass CustomSpecAlgo to override is_*() / create_worker."""

    def test_subclass_overrides_is_eagle(self):

        # EagleLike类
        class EagleLike(CustomSpecAlgo):

            # EagleLike类的is_eagle
            def is_eagle(self) -> bool:
                return True

        @SpeculativeAlgorithm.register(
            "MY_LIKE_EAGLE", supports_overlap=True, spec_class=EagleLike
        )

        # 内部方法_factory
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string("MY_LIKE_EAGLE")
        self.assertIsInstance(algo, EagleLike)
        self.assertIsInstance(algo, CustomSpecAlgo)
        self.assertTrue(algo.is_eagle())  # 断言为真
        # Other predicates default to False
        self.assertFalse(algo.is_ngram())  # 断言为假
        self.assertFalse(algo.is_dflash())  # 断言为假

    # TestSubclassOverride类的测试subclassoverridescreateworker
    def test_subclass_overrides_create_worker(self):

        # CustomDispatch类
        class CustomDispatch(CustomSpecAlgo):

            # CustomDispatch类的create_worker
            def create_worker(self, server_args):
                return "custom-dispatched"

        @SpeculativeAlgorithm.register("MY_CUSTOM", spec_class=CustomDispatch)

        # 内部方法_factory
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string("MY_CUSTOM")
        # Custom dispatch bypasses default overlap check
        self.assertEqual(algo.create_worker(MagicMock()), "custom-dispatched")  # 断言相等


# TestCrossTypeIdentity类
class TestCrossTypeIdentity(_RegistryIsolated):
    """A plugin algo and a builtin enum value must never compare equal."""

    def test_plugin_not_equal_to_builtin(self):
        @SpeculativeAlgorithm.register("MY_FOO")

        # 内部方法_factory
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string("MY_FOO")
        self.assertNotEqual(algo, SpeculativeAlgorithm.EAGLE)  # 断言不相等
        self.assertNotEqual(algo, SpeculativeAlgorithm.NONE)  # 断言不相等
        self.assertIsNot(algo, SpeculativeAlgorithm.EAGLE)  # 断言不是同一对象


if __name__ == "__main__":
    unittest.main(verbosity=3)
