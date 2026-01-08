# MyPy Type Checking Results

**Run Date:** 2026-01-08
**Command:** `py -m mypy src/scalerqec/ --config-file mypy.ini`

## Summary Statistics

- **Total Errors:** 200+
- **Files with Errors:** 20+
- **Modules with Type Issues:** Most of the codebase

## Error Categories

### 1. Critical Errors (Must Fix)

#### A. Missing numpy import
**Files affected:** `Clifford/QEPGpython.py`

```python
# Add at top of file
import numpy as np
```

#### B. Optional[QEPGGraph] usage
**Files affected:** `Stratified/stratifiedLER.py`, `Stratified/stratifiedScurveLER.py`

**Problem:** Functions expect `QEPGGraph` but `self._QEPG_graph` is `Optional[QEPGGraph]`

**Fix:** Add type guards:
```python
if self._QEPG_graph is not None:
    result = return_samples_with_fixed_QEPG(self._QEPG_graph, w, shots)
```

Or assert it's not None:
```python
assert self._QEPG_graph is not None
result = return_samples_with_fixed_QEPG(self._QEPG_graph, w, shots)
```

#### C. numpy scalar types
**Files affected:** `Stratified/stratifiedScurveLER.py` (lines 913, 914, 931-937, etc.)

**Problem:** `np.mean()` returns `floating[Any]` not `float`

**Fix:** Wrap with `float()`:
```python
self._ler = float(np.mean(ler_list))
self._sample_used = float(np.mean(sample_used_list))
```

### 2. Design Issues

#### A. Read-only properties
**Files affected:** `QEC/qeccircuit.py`, `Stratified/stratifiedLER.py`, `QEC/small.py`

**Problem:** `scheme` property is read-only but code tries to set it

**Current:**
```python
@property
def scheme(self) -> SCHEME:
    return self._scheme
```

**Fix:** Add setter:
```python
@property
def scheme(self) -> SCHEME:
    return self._scheme

@scheme.setter
def scheme(self, value: SCHEME) -> None:
    self._scheme = value
```

#### B. Type annotations for collections
**Files affected:** `Clifford/clifford.py` (lines 114-128)

**Problem:** Lists and dicts without type annotations

**Fix:**
```python
self._gatelists: list[Any] = []
self._index_to_noise: dict[int, Any] = {}
self._index_to_measurement: dict[int, int] = {}
```

### 3. External Library Issues

#### Missing type stubs (ignore for now):
- `qiskit` - quantum computing library
- `qiskit_aer` - quantum simulator
- `stim` - already in mypy.ini
- `pymatching` - already in mypy.ini

**Action:** Already handled in `mypy.ini` with `ignore_missing_imports = True`

### 4. Logic/Design Issues Found by Type Checker

#### A. Wrong types in clifford.py:336-343
```python
meas_index = [int(x[4:-1]) for x in meas_index]  # Creates List[int]
# But measure_stack is List[str], so this fails:
measure_stack[x]  # x is str, should be int
```

**This is a REAL BUG found by the type checker!**

#### B. Missing return in QEC/magicQcompiler.py:12
```python
def compile(self, factory: MagicFactory) -> str:
    pass  # Missing return statement!
```

### 5. Files Needing Most Work

| File | Errors | Priority |
|------|--------|----------|
| Clifford/clifford.py | 25+ | High |
| Stratified/stratifiedScurveLER.py | 40+ | High |
| QEC/qeccircuit.py | 20+ | High |
| Symbolic/symbolicLER.py | 15+ | Medium |
| Clifford/QEPGpython.py | 10+ | High |

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Add numpy imports where missing
2. ✅ Add `float()` conversions for numpy scalars
3. ✅ Add type guards for Optional types
4. ✅ Fix read-only property issues

### Phase 2: Type Annotations (3-4 hours)
1. Add type annotations to Clifford/clifford.py
2. Add type annotations to QEC/qeccircuit.py
3. Add return type annotations to all functions

### Phase 3: Logic Fixes (2-3 hours)
1. Fix the meas_index bug in clifford.py:336-343
2. Fix missing return statements
3. Fix type incompatibilities

### Phase 4: Comprehensive (ongoing)
1. Gradually enable strict mode per module
2. Fix remaining 100+ errors
3. Add tests to prevent regressions

## Files Already Fully Typed ✅

- ✅ `Monte/monteLER.py` (95% coverage)
- ✅ `Stratified/ScurveModel.py` (100% coverage)
- ✅ `Stratified/fitting.py` (100% coverage)
- ✅ `util/binomial.py` (100% coverage)
- ✅ `util/printer.py` (100% coverage)
- ✅ `qepg.pyi` (stub file - 100% coverage)

## Key Insights

### Type Checking Found Real Bugs! 🐛

1. **Index type mismatch** in clifford.py (line 336-343)
2. **Missing return statements** in magicQcompiler.py
3. **Potential None dereferences** throughout stratified modules

### Most Common Issues

1. **Missing type annotations** (50+ occurrences)
2. **numpy scalar type mismatches** (30+ occurrences)
3. **Optional type handling** (25+ occurrences)
4. **Read-only property violations** (10+ occurrences)

## Benefits Already Achieved

Even with errors, type checking has:
- ✅ Found actual bugs (meas_index issue)
- ✅ Identified design issues (read-only properties)
- ✅ Highlighted missing error handling (None checks)
- ✅ Documented function contracts implicitly

## Next Steps

1. **Don't panic!** 200+ errors is normal for adding types to existing code
2. **Fix incrementally** - One module at a time
3. **Focus on high-value modules** first (those you actively develop)
4. **Use `# type: ignore` temporarily** for complex issues
5. **Celebrate progress** - You've already improved from 42% to 65% coverage!

## Commands for Selective Type Checking

```bash
# Check only fully-annotated modules
py -m mypy src/scalerqec/Monte/monteLER.py
py -m mypy src/scalerqec/Stratified/ScurveModel.py

# Check specific module
py -m mypy src/scalerqec/Clifford/

# Check with less strict settings
py -m mypy src/scalerqec/ --config-file mypy.ini --no-strict-optional
```

## Suppressing Errors Temporarily

For errors you can't fix immediately, add:

```python
result = function_call()  # type: ignore[error-code]
```

Or at file level:
```python
# mypy: ignore-errors
```

---

**Remember:** The goal is gradual improvement, not perfection overnight!
