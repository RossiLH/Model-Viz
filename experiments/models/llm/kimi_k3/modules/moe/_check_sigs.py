"""Print signatures of all measure functions."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "tests/models/llm/kimi_k3/modules")
import inspect

import test_moe
import test_router_qb
import test_situ_glu
import test_norm

for mod_name, mod in [("test_moe", test_moe), ("test_router_qb", test_router_qb),
                       ("test_situ_glu", test_situ_glu), ("test_norm", test_norm)]:
    print(f"\n=== {mod_name} ===")
    for name in sorted(dir(mod)):
        if name.startswith("measure_"):
            fn = getattr(mod, name)
            sig = inspect.signature(fn)
            print(f"  {name}{sig}")
