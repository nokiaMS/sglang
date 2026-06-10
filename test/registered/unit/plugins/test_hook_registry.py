# 文件名: test_hook_registry.py - 钩子注册表
"""
Unit tests for the hook registry system.

Covers: basic hooks (AROUND/BEFORE/AFTER/REPLACE), descriptor preservation
(classmethod/staticmethod), hook ordering, cross-target conflict detection,
patch propagation, and edge cases.

Run:  python -m pytest test/registered/unit/plugins/test_hook_registry.py -v
"""

import sys
import types
import uuid

from sglang.srt.plugins.hook_registry import HookRegistry, HookType, plugin_hook
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

# ---------------------------------------------------------------------------
# Helpers: synthetic module creation
# ---------------------------------------------------------------------------

_SYNTH_MODULE_PREFIX = "_synth_hook_test_"


# 内部方法_make_module
def _make_module(**attrs):
    """Create a throwaway module registered in sys.modules."""
    name = f"{_SYNTH_MODULE_PREFIX}{uuid.uuid4().hex[:8]}"
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod, name


# 内部方法_cleanup_synth_modules
def _cleanup_synth_modules():
    """Remove all synthetic modules from sys.modules."""
    to_del = [k for k in sys.modules if k.startswith(_SYNTH_MODULE_PREFIX)]
    for k in to_del:
        del sys.modules[k]


# ---------------------------------------------------------------------------
# Base class for hook tests (shared setUp/tearDown)
# ---------------------------------------------------------------------------


class _HookTestCase(CustomTestCase):
    """Base class that resets HookRegistry and cleans up synth modules."""

    def setUp(self):
        HookRegistry.reset()
        _cleanup_synth_modules()

    # _HookTestCase类的测试清理
    def tearDown(self):
        HookRegistry.reset()
        _cleanup_synth_modules()


# ===========================================================================
# TestBasicHooks
# ===========================================================================


class TestBasicHooks(_HookTestCase):
    """AROUND / BEFORE / AFTER / REPLACE on plain functions, class REPLACE,
    and the @plugin_hook decorator."""

    # TestBasicHooks类的测试aroundfunction
    def test_around_function(self):

        # orig
        def orig(x):
            return x * 2

        mod, name = _make_module(orig=orig)

        # add_one
        def add_one(original_fn, x):
            return original_fn(x) + 1

        HookRegistry.register(f"{name}.orig", add_one, HookType.AROUND)
        HookRegistry.apply_hooks()
        self.assertEqual(mod.orig(3), 7)  # 3*2 + 1  # 断言相等

    # TestBasicHooks类的测试beforemodifiesargs
    def test_before_modifies_args(self):
        """BEFORE hook returns (args, kwargs) to modify arguments."""

        def orig(x, y=0):
            return x + y

        mod, name = _make_module(orig=orig)

        # double_x
        def double_x(x, y=0):
            return (x * 2,), {"y": y + 1}

        HookRegistry.register(f"{name}.orig", double_x, HookType.BEFORE)
        HookRegistry.apply_hooks()
        self.assertEqual(mod.orig(3), 7)  # x=3*2=6, y=0+1=1, 6+1=7  # 断言相等

    # TestBasicHooks类的测试beforereturningnone
    def test_before_returning_none(self):
        """BEFORE hook returning None leaves arguments unchanged."""

        def orig(x):
            return x * 2

        mod, name = _make_module(orig=orig)

        # before_noop
        def before_noop(x):
            return None  # leave args unchanged

        HookRegistry.register(f"{name}.orig", before_noop, HookType.BEFORE)
        HookRegistry.apply_hooks()
        self.assertEqual(mod.orig(3), 6)  # args unchanged  # 断言相等

    # TestBasicHooks类的测试afterfunction
    def test_after_function(self):

        # orig
        def orig(x):
            return x * 2

        mod, name = _make_module(orig=orig)

        # add_ten
        def add_ten(result, x):
            return result + 10

        HookRegistry.register(f"{name}.orig", add_ten, HookType.AFTER)
        HookRegistry.apply_hooks()
        self.assertEqual(mod.orig(3), 16)  # 3*2 + 10  # 断言相等

    # TestBasicHooks类的测试replacefunction
    def test_replace_function(self):

        # orig
        def orig(x):
            return x * 2

        mod, name = _make_module(orig=orig)

        # replacement
        def replacement(x):
            return x * 100

        HookRegistry.register(f"{name}.orig", replacement, HookType.REPLACE)
        HookRegistry.apply_hooks()
        self.assertEqual(mod.orig(3), 300)  # 断言相等

    # TestBasicHooks类的测试classreplace
    def test_class_replace(self):

        # Original类
        class Original:

            # Original类的greet
            def greet(self):
                return "original"

        mod, name = _make_module(Original=Original)

        # Replacement类
        class Replacement(Original):

            # Replacement类的greet
            def greet(self):
                return "replaced"

        HookRegistry.register(f"{name}.Original", Replacement, HookType.REPLACE)
        HookRegistry.apply_hooks()

        self.assertIs(mod.Original, Replacement)  # 断言是同一对象
        self.assertIsInstance(mod.Original(), Replacement)
        self.assertEqual(mod.Original().greet(), "replaced")  # 断言相等

    # TestBasicHooks类的测试pluginhookdecorator
    def test_plugin_hook_decorator(self):

        # orig
        def orig(x):
            return x

        mod, name = _make_module(orig=orig)

        @plugin_hook(f"{name}.orig", type=HookType.REPLACE)

        # my_replace
        def my_replace(x):
            return x + 42

        HookRegistry.apply_hooks()
        self.assertEqual(mod.orig(0), 42)  # 断言相等


# ===========================================================================
# TestDescriptorPreservation  (Bug B regression tests)
# ===========================================================================


class TestDescriptorPreservation(_HookTestCase):
    """Hooks on classmethod/staticmethod must preserve descriptor semantics."""

    def _make_cls_module(self):

        # MyClass类
        class MyClass:
            @classmethod

            # MyClass类的cm
            def cm(cls, x):
                return ("cm", cls.__name__, x)

            @staticmethod

            # MyClass类的sm
            def sm(x):
                return ("sm", x)

        mod, name = _make_module(MyClass=MyClass)
        return mod, name, MyClass

    # TestDescriptorPreservation类的测试aroundclassmethod
    def test_around_classmethod(self):
        mod, name, MyClass = self._make_cls_module()

        # add_tag
        def add_tag(original_fn, cls, x):
            return original_fn(cls, x) + ("around",)

        HookRegistry.register(f"{name}.MyClass.cm", add_tag, HookType.AROUND)
        HookRegistry.apply_hooks()

        result = mod.MyClass.cm(1)
        self.assertEqual(result, ("cm", "MyClass", 1, "around"))  # 断言相等

    # TestDescriptorPreservation类的测试replaceclassmethod
    def test_replace_classmethod(self):
        mod, name, MyClass = self._make_cls_module()

        # new_cm
        def new_cm(cls, x):
            return ("replaced_cm", cls.__name__, x)

        HookRegistry.register(f"{name}.MyClass.cm", new_cm, HookType.REPLACE)
        HookRegistry.apply_hooks()

        result = mod.MyClass.cm(1)
        self.assertEqual(result, ("replaced_cm", "MyClass", 1))  # 断言相等

    # TestDescriptorPreservation类的测试aroundstaticmethod
    def test_around_staticmethod(self):
        mod, name, MyClass = self._make_cls_module()

        # wrap_sm
        def wrap_sm(original_fn, x):
            return original_fn(x) + ("around",)

        HookRegistry.register(f"{name}.MyClass.sm", wrap_sm, HookType.AROUND)
        HookRegistry.apply_hooks()

        result = mod.MyClass.sm(1)
        self.assertEqual(result, ("sm", 1, "around"))  # 断言相等

    # TestDescriptorPreservation类的测试replacestaticmethod
    def test_replace_staticmethod(self):
        mod, name, MyClass = self._make_cls_module()

        # new_sm
        def new_sm(x):
            return ("replaced_sm", x)

        HookRegistry.register(f"{name}.MyClass.sm", new_sm, HookType.REPLACE)
        HookRegistry.apply_hooks()

        result = mod.MyClass.sm(1)
        self.assertEqual(result, ("replaced_sm", 1))  # 断言相等

    # TestDescriptorPreservation类的测试classmethodsubclasscls
    def test_classmethod_subclass_cls(self):
        mod, name, MyClass = self._make_cls_module()

        # add_tag
        def add_tag(original_fn, cls, x):
            return original_fn(cls, x) + ("around",)

        HookRegistry.register(f"{name}.MyClass.cm", add_tag, HookType.AROUND)
        HookRegistry.apply_hooks()

        # Sub类
        class Sub(mod.MyClass):
            pass

        result = Sub.cm(1)
        self.assertEqual(result, ("cm", "Sub", 1, "around"))  # 断言相等


# ===========================================================================
# TestHookOrdering
# ===========================================================================


class TestHookOrdering(_HookTestCase):
    """Verify REPLACE is applied first, then other hooks wrap it."""

    def test_replace_then_around(self):

        # orig
        def orig(x):
            return x

        mod, name = _make_module(orig=orig)

        # repl
        def repl(x):
            return x * 10

        # add_one
        def add_one(original_fn, x):
            return original_fn(x) + 1

        HookRegistry.register(f"{name}.orig", repl, HookType.REPLACE)
        HookRegistry.register(f"{name}.orig", add_one, HookType.AROUND)
        HookRegistry.apply_hooks()
        # REPLACE first: x*10, then AROUND: +1  => 31
        self.assertEqual(mod.orig(3), 31)  # 断言相等

    # TestHookOrdering类的测试replacebeforeafter
    def test_replace_before_after(self):

        # orig
        def orig(x):
            return x

        mod, name = _make_module(orig=orig)

        # repl
        def repl(x):
            return x * 10

        # double_arg
        def double_arg(x):
            return (x * 2,), {}

        # add_hundred
        def add_hundred(result, x):
            return result + 100

        HookRegistry.register(f"{name}.orig", repl, HookType.REPLACE)
        HookRegistry.register(f"{name}.orig", double_arg, HookType.BEFORE)
        HookRegistry.register(f"{name}.orig", add_hundred, HookType.AFTER)
        HookRegistry.apply_hooks()
        # BEFORE doubles x: 3*2=6 → REPLACE: 6*10=60 → AFTER: 60+100=160
        self.assertEqual(mod.orig(3), 160)  # 断言相等


# ===========================================================================
# TestCrossTargetConflict
# ===========================================================================


class TestCrossTargetConflict(_HookTestCase):
    """Verify warning for class REPLACE + method REPLACE combo."""

    def test_class_replace_then_method_replace_warns(self):

        # Original类
        class Original:

            # Original类的foo
            def foo(self):
                return "orig"

        mod, name = _make_module(Original=Original)

        # Replacement类
        class Replacement(Original):

            # Replacement类的foo
            def foo(self):
                return "class_replaced"

        HookRegistry.register(f"{name}.Original", Replacement, HookType.REPLACE)

        # method_repl
        def method_repl(self):
            return "method_replaced"

        HookRegistry.register(f"{name}.Original.foo", method_repl, HookType.REPLACE)

        with self.assertLogs("sglang.srt.plugins.hook_registry", level="WARNING") as cm:
            HookRegistry.apply_hooks()

        self.assertTrue(any("will override" in msg for msg in cm.output))  # 断言为真


# ===========================================================================
# TestPatchPropagation
# ===========================================================================


class TestPatchPropagation(_HookTestCase):
    """Verify that patches propagate to other modules that imported the target."""

    def test_same_reference_propagates(self):

        # orig
        def orig(x):
            return x * 2

        source_mod, source_name = _make_module(orig=orig)
        importer_mod, _ = _make_module(orig=orig)  # same reference

        # add_one
        def add_one(fn, x):
            return fn(x) + 1

        HookRegistry.register(f"{source_name}.orig", add_one, HookType.AROUND)
        HookRegistry.apply_hooks()

        self.assertEqual(source_mod.orig(3), 7)  # 断言相等
        self.assertEqual(importer_mod.orig(3), 7)  # 断言相等


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases(_HookTestCase):
    """Reset, type validation, multi-AROUND onion, idempotent apply."""

    def test_reset(self):

        # orig
        def orig(x):
            return x

        mod, name = _make_module(orig=orig)

        # noop
        def noop(fn, x):
            return fn(x)

        HookRegistry.register(f"{name}.orig", noop, HookType.AROUND)
        HookRegistry.reset()

        HookRegistry.apply_hooks()
        self.assertEqual(mod.orig(3), 3)  # 断言相等

    # TestEdgeCases类的测试registerclasswithwrongtype
    def test_register_class_with_wrong_type(self):

        # BadHook类
        class BadHook:
            pass

        for ht in (HookType.BEFORE, HookType.AFTER, HookType.AROUND):
            with self.assertRaises(TypeError):  # 断言抛出异常
                HookRegistry.register("some.target", BadHook, ht)

    # TestEdgeCases类的测试multiaroundonion
    def test_multi_around_onion(self):
        call_order = []

        # orig
        def orig(x):
            call_order.append("orig")
            return x

        mod, name = _make_module(orig=orig)

        # around1
        def around1(fn, x):
            call_order.append("a1_before")
            result = fn(x)
            call_order.append("a1_after")
            return result + 1

        # around2
        def around2(fn, x):
            call_order.append("a2_before")
            result = fn(x)
            call_order.append("a2_after")
            return result + 10

        HookRegistry.register(f"{name}.orig", around1, HookType.AROUND)
        HookRegistry.register(f"{name}.orig", around2, HookType.AROUND)
        HookRegistry.apply_hooks()

        result = mod.orig(0)
        self.assertEqual(result, 11)  # 断言相等
        self.assertEqual(  # 断言相等
            call_order, ["a2_before", "a1_before", "orig", "a1_after", "a2_after"]
        )

    # TestEdgeCases类的测试applyidempotent
    def test_apply_idempotent(self):
        call_count = [0]

        # orig
        def orig(x):
            return x

        mod, name = _make_module(orig=orig)

        # counter
        def counter(fn, x):
            call_count[0] += 1
            return fn(x)

        HookRegistry.register(f"{name}.orig", counter, HookType.AROUND)
        HookRegistry.apply_hooks()
        HookRegistry.apply_hooks()  # second apply should be no-op

        mod.orig(1)
        self.assertEqual(call_count[0], 1)  # 断言相等


if __name__ == "__main__":
    import unittest

    unittest.main()
