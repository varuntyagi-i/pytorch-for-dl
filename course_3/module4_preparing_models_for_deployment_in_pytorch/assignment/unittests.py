# Unitest for c3m4_assignment

from types import FunctionType
from dlai_grader.grading import print_feedback, test_case

import torch
import torch.nn as nn

def exercise1(learner_func):
    def g():
        cases = []

        # === function type check ===
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "prune_model has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        # === end function type check ===

        try:
            # Small toy model with both Conv2d and Linear
            class TinyNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Make out_channels = 10 to test structured pruning channel counts
                    self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1, bias=False)
                    self.relu = nn.ReLU()
                    self.pool = nn.AdaptiveAvgPool2d((4, 4))
                    self.flatten = nn.Flatten()
                    self.fc = nn.Linear(10 * 4 * 4, 5, bias=False)

                def forward(self, x):
                    x = self.pool(self.relu(self.conv(x)))
                    x = self.flatten(x)
                    x = self.fc(x)
                    return x

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x = torch.randn(2, 3, 16, 16, device=device)

            # Helper: get prunable modules
            def prunable_modules(m):
                for _, mod in m.named_modules():
                    if isinstance(mod, (nn.Conv2d, nn.Linear)):
                        yield mod

            # ---------- Case 1: l1_unstructured, amount=0.0 (no-op mask of ones) ----------
            model = TinyNet().to(device).eval()
            out_before = model(x).detach()
            ret = learner_func(model, amount=0.0, mode="l1_unstructured")

            # Same instance?
            t = test_case()
            if ret is not model:
                t.failed = True
                t.msg = "Function must prune in-place and return the SAME model instance."
                t.want = "Return original instance"
                t.got = "Returned a different object"
            cases.append(t)

            # Reparam attrs exist and mask is all ones; weight == weight_orig
            for mod in prunable_modules(model):
                t = test_case()
                has_attrs = hasattr(mod, "weight_orig") and hasattr(mod, "weight_mask")
                if not has_attrs:
                    t.failed = True
                    t.msg = "Missing pruning reparam attributes after amount=0.0"
                    t.want = "weight_orig and weight_mask present"
                    t.got = f"has weight_orig? {hasattr(mod,'weight_orig')}, weight_mask? {hasattr(mod,'weight_mask')}"
                else:
                    mask = mod.weight_mask.detach()
                    if mask.dtype != torch.float32 and mask.dtype != torch.float64:
                        mask = mask.float()
                    all_ones = torch.allclose(mask, torch.ones_like(mask))
                    if not all_ones:
                        t.failed = True
                        t.msg = "Mask should be all ones for amount=0.0 (no weights pruned)."
                        t.want = "All ones mask"
                        t.got = f"mask sum={mask.sum().item():.0f} / {mask.numel()}"
                    else:
                        # weight should equal weight_orig when mask is ones
                        if not torch.allclose(mod.weight.detach(), mod.weight_orig.detach()):
                            t.failed = True
                            t.msg = "With mask=1, weight must equal weight_orig."
                            t.want = "weight == weight_orig"
                            t.got = "Different tensors"
                cases.append(t)

            # Forward still works & output shape unchanged
            t = test_case()
            out_after = model(x).detach()
            if out_before.shape != out_after.shape:
                t.failed = True
                t.msg = "Pruning must not change tensor shapes."
                t.want = f"Output shape {tuple(out_before.shape)}"
                t.got = f"{tuple(out_after.shape)}"
            cases.append(t)

            # ---------- Case 2: l1_unstructured, amount=1.0 (all zero) ----------
            model = TinyNet().to(device).eval()
            learner_func(model, amount=1.0, mode="l1_unstructured")

            for mod in prunable_modules(model):
                t = test_case()
                if not (hasattr(mod, "weight_orig") and hasattr(mod, "weight_mask")):
                    t.failed = True
                    t.msg = "Missing pruning reparam attributes after amount=1.0"
                    t.want = "weight_orig and weight_mask present"
                    t.got = f"has weight_orig? {hasattr(mod,'weight_orig')}, weight_mask? {hasattr(mod,'weight_mask')}"
                else:
                    mask = mod.weight_mask.detach()
                    zeros = torch.count_nonzero(mask).item() == 0
                    all_zero_weight = torch.count_nonzero(mod.weight.detach()).item() == 0
                    if not zeros or not all_zero_weight:
                        t.failed = True
                        t.msg = "All weights should be zero with amount=1.0 (unstructured)."
                        t.want = "All-zero mask and weight"
                        t.got = f"mask_nonzero={int(torch.count_nonzero(mask))}, weight_nonzero={int(torch.count_nonzero(mod.weight.detach()))}"
                cases.append(t)

            # Forward still runs
            t = test_case()
            try:
                _ = model(x)
            except Exception as e:
                t.failed = True
                t.msg = f"Forward failed after pruning all weights: {e}"
                t.want = "Forward pass succeeds"
                t.got = "Exception"
            cases.append(t)

            # ---------- Case 3: ln_structured (channel pruning) ----------
            model = TinyNet().to(device).eval()
            learner_func(model, amount=0.3, mode="ln_structured")
            conv = model.conv
            fc = model.fc

            # For structured pruning, entire output channels (dim=0) should be zeroed.
            def count_zeroed_channels(weight_mask):
                # weight_mask shape: (out_channels, ...)
                with torch.no_grad():
                    flat = weight_mask.view(weight_mask.size(0), -1)
                    zero_rows = (flat.abs().sum(dim=1) == 0).sum().item()
                return zero_rows

            # Conv2d channel count
            t = test_case()
            if not hasattr(conv, "weight_mask"):
                t.failed = True
                t.msg = "Conv layer missing weight_mask after ln_structured."
                t.want = "weight_mask present"
                t.got = "Absent"
            else:
                zc = count_zeroed_channels(conv.weight_mask.detach())
                # Expect roughly 30% of 10 ≈ 3 channels (allow ±1 tolerance)
                if not (2 <= zc <= 4):
                    t.failed = True
                    t.msg = "Unexpected number of zeroed output channels in Conv2d for 30% structured pruning."
                    t.want = "≈3 (±1) zeroed channels"
                    t.got = f"{zc} zeroed channels"
            cases.append(t)

            # Linear channel (out_features) pruning: same logic
            t = test_case()
            if not hasattr(fc, "weight_mask"):
                t.failed = True
                t.msg = "Linear layer missing weight_mask after ln_structured."
                t.want = "weight_mask present"
                t.got = "Absent"
            else:
                zc = count_zeroed_channels(fc.weight_mask.detach())
                # 30% of 5 ≈ 1–2 channels
                if not (1 <= zc <= 2):
                    t.failed = True
                    t.msg = "Unexpected number of zeroed output channels in Linear for 30% structured pruning."
                    t.want = "≈1–2 zeroed channels"
                    t.got = f"{zc} zeroed channels"
            cases.append(t)

            # Forward still runs
            t = test_case()
            try:
                _ = model(x)
            except Exception as e:
                t.failed = True
                t.msg = f"Forward failed after ln_structured pruning: {e}"
                t.want = "Forward pass succeeds"
                t.got = "Exception"
            cases.append(t)

            # ---------- Case 4: invalid mode → ValueError ----------
            model = TinyNet().to(device).eval()
            t = test_case()
            try:
                learner_func(model, amount=0.2, mode="not_a_mode")
                t.failed = True
                t.msg = "Expected ValueError for invalid mode, but none was raised."
                t.want = "Raise ValueError"
                t.got = "No error"
            except ValueError:
                pass
            except Exception as e:
                t.failed = True
                t.msg = f"Wrong exception type for invalid mode: {e}"
                t.want = "ValueError"
                t.got = type(e)
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error while executing tests: {e}"
            t.want = "Execution without errors"
            t.got = "Exception raised"
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

def exercise2(learner_func):
    def g():
        cases = []

        # === general checks ===
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "quantize_dynamic_linear has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        try:
            # --- tiny test model: Conv (should stay fp32) + two Linear (should be dynamically quantized) ---
            model = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(8 * 4 * 4, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
            )
            model.train()  # keep original in train mode to detect unwanted mutation

            # pre-call snapshots
            orig_id = id(model)
            orig_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
            orig_conv_id = id(orig_conv)
            orig_lin_count = sum(isinstance(m, nn.Linear) for m in model.modules())
            orig_state = {k: v.clone() for k, v in model.state_dict().items()}

            # run learner function
            qmodel = learner_func(model)

            # --- 1) returns a model ---
            t = test_case()
            if not isinstance(qmodel, nn.Module):
                t.failed = True
                t.msg = "Function should return an nn.Module."
                t.want = nn.Module
                t.got = type(qmodel)
            cases.append(t)

            # --- 2) new copy was made; original not mutated (mode + identity + params preserved) ---
            t = test_case()
            if qmodel is model or id(qmodel) == orig_id:
                t.failed = True
                t.msg = "Returned model must be a NEW copy (not the same object as input)."
                t.want = "Deep-copied model"
                t.got = "Same object"
            cases.append(t)

            t = test_case()
            if model.training is False:
                t.failed = True
                t.msg = "Input model's training/eval mode was mutated."
                t.want = "Original model remains in train() mode"
                t.got = "model.eval() was applied to the original"
            cases.append(t)

            t = test_case()
            # original weights unchanged
            changed = []
            for k, v in model.state_dict().items():
                if k in orig_state and not torch.equal(v, orig_state[k]):
                    changed.append(k)
            if changed:
                t.failed = True
                t.msg = f"Original model parameters were modified: {changed[:5]}..."
                t.want = "No mutation of input model"
                t.got = "Weights changed"
            cases.append(t)

            t = test_case()
            # even un-quantized modules (Conv2d) should be a different object in the copy
            q_conv = next(m for m in qmodel.modules() if isinstance(m, nn.Conv2d))
            if id(q_conv) == orig_conv_id:
                t.failed = True
                t.msg = "Deepcopy not used: Conv2d module object is shared between original and returned model."
                t.want = "Different Conv2d object ids"
                t.got = f"Same id: {orig_conv_id}"
            cases.append(t)

            # --- 3) eval() mode on the returned model ---
            t = test_case()
            if qmodel.training:
                t.failed = True
                t.msg = "Returned model should be in eval() mode."
                t.want = "eval()"
                t.got = "train()"
            cases.append(t)

            # --- 4) ONLY Linear layers are dynamically quantized ---
            # count dynamic Linear modules in quantized model
            q_dyn_linear_type = getattr(nn.quantized.dynamic, "Linear", None)
            q_dyn_linear_count = sum(isinstance(m, q_dyn_linear_type) for m in qmodel.modules())
            q_fp32_linear_count = sum(isinstance(m, nn.Linear) for m in qmodel.modules())
            q_conv_count = sum(isinstance(m, nn.Conv2d) for m in qmodel.modules())
            orig_conv_count = sum(isinstance(m, nn.Conv2d) for m in model.modules())

            t = test_case()
            if q_dyn_linear_count != orig_lin_count:
                t.failed = True
                t.msg = "All nn.Linear layers must be dynamically quantized."
                t.want = f"{orig_lin_count} dynamically quantized Linear layer(s)"
                t.got = f"{q_dyn_linear_count} dynamically quantized Linear layer(s)"
            cases.append(t)

            t = test_case()
            if q_fp32_linear_count != 0:
                t.failed = True
                t.msg = "No nn.Linear layers should remain in the returned model."
                t.want = "0 fp32 Linear layers"
                t.got = f"{q_fp32_linear_count} fp32 Linear layers still present"
            cases.append(t)

            t = test_case()
            if q_conv_count != orig_conv_count:
                t.failed = True
                t.msg = "Non-Linear modules (e.g., Conv2d) must not be quantized/removed."
                t.want = f"{orig_conv_count} Conv2d layer(s)"
                t.got = f"{q_conv_count} Conv2d layer(s)"
            cases.append(t)


            # --- 5) Right parameters for dynamic quantization: dtype matches backend
            t = test_case()

            # get dynamic Linear class from either namespace
            q_dyn_linear_type = getattr(nn.quantized.dynamic, "Linear", None)
            if q_dyn_linear_type is None:
                try:
                    from torch.ao.nn.quantized.dynamic.modules.linear import Linear as QDynLinear
                    q_dyn_linear_type = QDynLinear
                except Exception:
                    q_dyn_linear_type = None

            # if we can't resolve the class, we can't do this check at all
            if q_dyn_linear_type is None:
                # Skip this check (no failure). Older/newer PyTorch layout.
                cases.append(t)
            else:
                # determine expected dtype from engine
                engine = getattr(getattr(torch.backends, "quantized", object), "engine", "")
                expected = torch.qint8 if engine == "fbgemm" else torch.quint8

                def _probe_dtype(mod):
                    # Try common attributes
                    for attr in ("dtype", "weight_dtype", "_dtype", "_weight_dtype"):
                        dt = getattr(mod, attr, None)
                        if dt is not None:
                            return dt
                    # Try packed params object (implementation detail; may not exist)
                    pp = getattr(mod, "_packed_params", None)
                    if pp is not None:
                        for attr in ("dtype", "weight_dtype", "_dtype", "_weight_dtype"):
                            dt = getattr(pp, attr, None)
                            if dt is not None:
                                return dt
                    # As a last resort, sniff repr() for qint8/quint8 hints
                    r = repr(mod)
                    if "qint8" in r:
                        return torch.qint8
                    if "quint8" in r:
                        return torch.quint8
                    return None  # unknown on this build

                wrong, unknown = [], []
                for m in qmodel.modules():
                    if isinstance(m, q_dyn_linear_type):
                        dt = _probe_dtype(m)
                        if dt is None:
                            unknown.append(type(m).__name__)
                        elif dt != expected:
                            wrong.append((type(m).__name__, dt, engine))

                if wrong:
                    t.failed = True
                    t.msg = "Quantized Linear layers must use the correct INT8 dtype for the active backend."
                    t.want = f"{expected} when engine='{engine}'"
                    t.got = f"Mismatches: {wrong}"
                # If dtype is unknown (no attribute exposed), do NOT fail the test — module is quantized
                cases.append(t)

            # --- 6) CPU-only & forward works ---
            t = test_case()
            if any(p.device.type != "cpu" for p in qmodel.parameters()):
                t.failed = True
                t.msg = "Quantized model must run on CPU-only."
                t.want = "All parameters on CPU"
                t.got = "Some parameters on CUDA"
            cases.append(t)

            t = test_case()
            try:
                x = torch.randn(1, 3, 32, 32)  # CPU
                with torch.inference_mode():
                    y = qmodel(x)
                if not (isinstance(y, torch.Tensor) and y.shape[-1] == 2):
                    raise RuntimeError(f"Unexpected output: {type(y)}, shape={getattr(y, 'shape', None)}")
            except Exception as e:
                t.failed = True
                t.msg = f"Quantized model forward pass failed on CPU: {e}"
                t.want = "Forward pass runs on CPU"
                t.got = "Exception raised"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error while executing tests: {e}"
            t.want = "Execution without errors"
            t.got = "Exception raised"
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

def _try_fuse(seq: nn.Sequential, names):
    """
    Best-effort wrapper around torch.ao.quantization.fuse_modules.
    Replaces fused positions with intrinsic fused ops / Identity in-place.
    """
    try:
        fuse_modules = getattr(torch.ao.quantization, "fuse_modules", None)
        if fuse_modules is None:  # PyTorch < 1.13 fallback
            fuse_modules = getattr(torch.quantization, "fuse_modules")
        # Inplace fusion inside the *same* Sequential
        fuse_modules(seq, names, inplace=True)
    except Exception:
        # Best-effort: ignore unsupported patterns / backends
        pass

def exercise3(learner_func):
    def g():
        cases = []

        # === general checks ===
        from types import FunctionType
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "fuse_model_inplace has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        # === end general checks ===

        try:
            # ---------------- Tiny model with multiple fusible patterns ----------------
            class Nested(nn.Module):
                def __init__(self, ch=8):
                    super().__init__()
                    # Conv + BN + ReLU (should fuse by recursion)
                    self.seq = nn.Sequential(
                        nn.Conv2d(ch, ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(),
                    )

                def forward(self, x):
                    return self.seq(x)

            class TinyFuseNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 1) Conv + BN + ReLU (triplet)
                    self.stem = nn.Sequential(
                        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(8),
                        nn.ReLU(),
                    )
                    # 2) Conv + ReLU (pair)
                    self.body = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
                        nn.ReLU(),
                    )
                    # 3) Conv + BN (pair)
                    self.bn_only = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(8),
                    )
                    # 4) Unsupported pair (Conv + Sigmoid) → must remain untouched
                    self.tail = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=1, bias=False),
                        nn.Sigmoid(),
                    )
                    # 5) Linear + ReLU (pair) inside a classifier head
                    self.pool = nn.AdaptiveAvgPool2d((4, 4))
                    self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(8 * 4 * 4, 16, bias=False),
                        nn.ReLU(),
                        nn.Linear(16, 3, bias=False),
                    )
                    # 6) Not in nn.Sequential → must **not** fuse
                    self.nonseq_conv = nn.Conv2d(3, 3, kernel_size=1, bias=False)
                    self.nonseq_relu = nn.ReLU()

                    # 7) Nested module to test recursion
                    self.nested = Nested(ch=8)

                def forward(self, x):
                    x = self.stem(x)
                    x = self.body(x)
                    x = self.bn_only(x)
                    x = self.tail(x)
                    x = self.pool(x)
                    x = self.classifier(x)
                    return x

            model = TinyFuseNet().eval()

            # Snapshot counts/types before fusion
            bn_count_before = sum(isinstance(m, nn.BatchNorm2d) for m in model.modules())
            relu_count_before = sum(isinstance(m, nn.ReLU) for m in model.modules())
            orig_id = id(model)

            # Forward works before fusion
            t = test_case()
            try:
                with torch.inference_mode():
                    y0 = model(torch.randn(2, 3, 16, 16))
                if not (isinstance(y0, torch.Tensor) and y0.shape[-1] == 3):
                    raise RuntimeError("Unexpected output shape/type before fusion")
            except Exception as e:
                t.failed = True
                t.msg = f"Forward failed before fusion: {e}"
                t.want = "Forward pass succeeds"
                t.got = "Exception"
            cases.append(t)

            # ---------------- Run learner function (in-place) ----------------
            ret = learner_func(model)

            # Must be same object (in-place)
            t = test_case()
            if ret is not model or id(ret) != orig_id:
                t.failed = True
                t.msg = "Function must modify the model in-place and return the SAME (id not equal) instance (not the same object as input)."
                t.want = "Return original instance"
                t.got = "Returned a different object"
            cases.append(t)

            # Forward still works after fusion (best-effort, shapes preserved)
            t = test_case()
            try:
                with torch.inference_mode():
                    y1 = model(torch.randn(2, 3, 16, 16))
                if not (isinstance(y1, torch.Tensor) and y1.shape[-1] == 3):
                    raise RuntimeError("Unexpected output shape/type after fusion")
            except Exception as e:
                t.failed = True
                t.msg = f"Forward failed after fusion: {e}"
                t.want = "Forward pass succeeds"
                t.got = "Exception"
            cases.append(t)

            # BatchNorms in fusible pairs should be folded away (often replaced by Identity)
            bn_count_after = sum(isinstance(m, nn.BatchNorm2d) for m in model.modules())
            t = test_case()
            if bn_count_after >= bn_count_before:
                t.failed = True
                t.msg = "Expected fewer (or zero) BatchNorm2d modules after fusion."
                t.want = f"< {bn_count_before} BatchNorm2d modules"
                t.got = f"{bn_count_after} BatchNorm2d modules"
            cases.append(t)

            # ---- Targeted structure checks (Identity placeholders) ----
            # 1) Conv + BN + ReLU → BN and ReLU positions should become Identity
            t = test_case()
            stem_ok = isinstance(model.stem[1], nn.Identity) and isinstance(model.stem[2], nn.Identity)
            if not stem_ok:
                t.failed = True
                t.msg = "Conv+BN+ReLU in `stem` was not fused (expected BN and ReLU replaced by Identity)."
                t.want = "stem[1] and stem[2] are nn.Identity"
                t.got = f"stem types: {[type(m).__name__ for m in model.stem]}"
            cases.append(t)

            # 2) Conv + ReLU → ReLU position should become Identity
            t = test_case()
            body_ok = isinstance(model.body[1], nn.Identity)
            if not body_ok:
                t.failed = True
                t.msg = "Conv+ReLU in `body` was not fused (expected ReLU replaced by Identity)."
                t.want = "body[1] is nn.Identity"
                t.got = f"type(body[1])={type(model.body[1]).__name__}"
            cases.append(t)

            # 3) Conv + BN → BN position should become Identity
            t = test_case()
            bn_only_ok = isinstance(model.bn_only[1], nn.Identity)
            if not bn_only_ok:
                t.failed = True
                t.msg = "Conv+BN in `bn_only` was not fused (expected BN replaced by Identity)."
                t.want = "bn_only[1] is nn.Identity"
                t.got = f"type(bn_only[1])={type(model.bn_only[1]).__name__}"
            cases.append(t)

            # 4) Linear + ReLU → ReLU position should become Identity
            t = test_case()
            cls_ok = isinstance(model.classifier[2], nn.Identity)
            if not cls_ok:
                t.failed = True
                t.msg = "Linear+ReLU in `classifier` was not fused (expected ReLU replaced by Identity)."
                t.want = "classifier[2] is nn.Identity"
                t.got = f"type(classifier[2])={type(model.classifier[2]).__name__}"
            cases.append(t)

            # 5) Unsupported pattern (Conv + Sigmoid) must remain untouched
            t = test_case()
            tail_ok = isinstance(model.tail[1], nn.Sigmoid)
            if not tail_ok:
                t.failed = True
                t.msg = "Unsupported pattern (Conv+Sigmoid) should be left untouched."
                t.want = "tail[1] is nn.Sigmoid"
                t.got = f"type(tail[1])={type(model.tail[1]).__name__}"
            cases.append(t)

            # 6) Not in nn.Sequential → must NOT fuse
            t = test_case()
            if isinstance(model.nonseq_relu, nn.Identity):
                t.failed = True
                t.msg = "Layers not inside nn.Sequential should not be fused."
                t.want = "nonseq_relu remains nn.ReLU"
                t.got = "nn.Identity"
            cases.append(t)

            # 7) Recursion: nested.seq Conv+BN+ReLU fused (BN/ReLU → Identity)
            t = test_case()
            nested_ok = (
                isinstance(model.nested.seq[1], nn.Identity) and
                isinstance(model.nested.seq[2], nn.Identity)
            )
            if not nested_ok:
                t.failed = True
                t.msg = "Recursive fusion failed in nested module."
                t.want = "nested.seq[1] and nested.seq[2] are nn.Identity"
                t.got = f"nested.seq types: {[type(m).__name__ for m in model.nested.seq]}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error while executing tests: {e}"
            t.want = "Execution without errors"
            t.got = "Exception raised"
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise4(learner_func):
    def g():
        cases = []

        # === general checks ===
        from types import FunctionType
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "prepare_qat has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        # === end general checks ===

        try:
            import torch
            import torch.nn as nn
            import torch.ao.quantization as aoq

            # ---------------- Tiny net with multiple fusible patterns + nesting ----------------
            class Nested(nn.Module):
                def __init__(self, ch=8):
                    super().__init__()
                    self.seq = nn.Sequential(
                        nn.Conv2d(ch, ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(),
                    )

                def forward(self, x): return self.seq(x)

            class TinyQATNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 1) Conv + BN + ReLU
                    self.stem = nn.Sequential(
                        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(8),
                        nn.ReLU(),
                    )
                    # 2) Conv + ReLU
                    self.body = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
                        nn.ReLU(),
                    )
                    # 3) Conv + BN
                    self.bn_only = nn.Sequential(
                        nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(8),
                    )
                    # 4) Classifier with Linear + ReLU
                    self.pool = nn.AdaptiveAvgPool2d((4, 4))
                    self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(8 * 4 * 4, 16, bias=False),
                        nn.ReLU(),
                        nn.Linear(16, 3, bias=False),
                    )
                    # 5) Nested (to verify recursion happened in fusion)
                    self.nested = Nested(8)

                def forward(self, x):
                    x = self.stem(x)
                    x = self.body(x)
                    x = self.bn_only(x)
                    x = self.pool(x)
                    x = self.classifier(x)
                    return x

            model = TinyQATNet()
            model.eval()  # keep original in eval() to detect unwanted mutation

            # ---------- Snapshots of original (to check "no mutation") ----------
            orig_id = id(model)
            orig_state = {k: v.clone() for k, v in model.state_dict().items()}
            orig_training = model.training

            # for deepcopy/object identity checks
            def get_first_conv(m):
                return next(x for x in m.modules() if isinstance(x, nn.Conv2d))
            orig_first_conv_id = id(get_first_conv(model))

            # Structural checks pre-QAT (these must remain in original)
            def original_structure_ok(m):
                return (
                    isinstance(m.stem[1], nn.BatchNorm2d) and isinstance(m.stem[2], nn.ReLU) and
                    isinstance(m.body[1], nn.ReLU) and
                    isinstance(m.bn_only[1], nn.BatchNorm2d) and
                    isinstance(m.classifier[2], nn.ReLU) and
                    isinstance(m.nested.seq[1], nn.BatchNorm2d) and
                    isinstance(m.nested.seq[2], nn.ReLU)
                )
            assert original_structure_ok(model), "Test setup invalid: original structure unexpected."

            # ---------- Call learner function ----------
            backend = "fbgemm"
            qat_model = learner_func(model, backend=backend)

            # --- 1) Returns a module, and is a NEW copy (deepcopy) ---
            t = test_case()
            if not isinstance(qat_model, nn.Module):
                t.failed = True
                t.msg = "Function should return an nn.Module."
                t.want = nn.Module
                t.got = type(qat_model)
            cases.append(t)

            t = test_case()
            if qat_model is model or id(qat_model) == orig_id:
                t.failed = True
                t.msg = "Returned model must be a NEW copy (do not mutate input)."
                t.want = "Deep-copied model"
                t.got = "Same object as input"
            cases.append(t)

            # original mode must be preserved (we set eval() above)
            t = test_case()
            if model.training != orig_training:
                t.failed = True
                t.msg = "Input model's train/eval mode was mutated."
                t.want = f"Original model remains {'train' if orig_training else 'eval'}()"
                t.got = f"{'train' if model.training else 'eval'}()"
            cases.append(t)

            # parameters unchanged in original
            t = test_case()
            changed = [k for k, v in model.state_dict().items() if not torch.equal(v, orig_state[k])]
            if changed:
                t.failed = True
                t.msg = f"Original model parameters were modified: {changed[:5]}..."
                t.want = "No mutation of input model weights"
                t.got = "Weights changed"
            cases.append(t)

            # deepcopy sanity: first Conv must be a different object id
            t = test_case()
            qat_first_conv_id = id(get_first_conv(qat_model))
            if qat_first_conv_id == orig_first_conv_id:
                t.failed = True
                t.msg = "Deepcopy not used: modules are shared between original and returned model."
                t.want = "Different module object ids"
                t.got = f"Same id {orig_first_conv_id}"
            cases.append(t)

            # --- 2) Returned model should be in train() mode ---
            t = test_case()
            if not qat_model.training:
                t.failed = True
                t.msg = "QAT-ready model must be returned in train() mode."
                t.want = "train()"
                t.got = "eval()"
            cases.append(t)

            # --- 3) Fusion should have been applied on the COPY (best-effort) ---
            # Expect BN/ReLU positions turned to Identity in fused sequences
            t = test_case()
            stem_fused = isinstance(qat_model.stem[1], nn.Identity) and isinstance(qat_model.stem[2], nn.Identity)
            if not stem_fused:
                t.failed = True
                t.msg = "Conv+BN+ReLU in `stem` was not fused on the prepared model."
                t.want = "qat_model.stem[1] and stem[2] are nn.Identity"
                t.got = f"stem types: {[type(m).__name__ for m in qat_model.stem]}"
            cases.append(t)

            t = test_case()
            body_fused = isinstance(qat_model.body[1], nn.Identity)
            if not body_fused:
                t.failed = True
                t.msg = "Conv+ReLU in `body` was not fused on the prepared model."
                t.want = "qat_model.body[1] is nn.Identity"
                t.got = type(qat_model.body[1]).__name__
            cases.append(t)

            t = test_case()
            bn_only_fused = isinstance(qat_model.bn_only[1], nn.Identity)
            if not bn_only_fused:
                t.failed = True
                t.msg = "Conv+BN in `bn_only` was not fused on the prepared model."
                t.want = "qat_model.bn_only[1] is nn.Identity"
                t.got = type(qat_model.bn_only[1]).__name__
            cases.append(t)

            t = test_case()
            linrelu_fused = isinstance(qat_model.classifier[2], nn.Identity)
            if not linrelu_fused:
                t.failed = True
                t.msg = "Linear+ReLU in `classifier` was not fused on the prepared model."
                t.want = "qat_model.classifier[2] is nn.Identity"
                t.got = type(qat_model.classifier[2]).__name__
            cases.append(t)

            t = test_case()
            nested_fused = (
                isinstance(qat_model.nested.seq[1], nn.Identity) and
                isinstance(qat_model.nested.seq[2], nn.Identity)
            )
            if not nested_fused:
                t.failed = True
                t.msg = "Recursive fusion failed in nested module on the prepared model."
                t.want = "qat_model.nested.seq[1] and [2] are nn.Identity"
                t.got = f"nested.seq types: {[type(m).__name__ for m in qat_model.nested.seq]}"
            cases.append(t)

            # --- 4) qconfig attached (backend-aware or sensible default) ---
            t = test_case()
            qc = getattr(qat_model, "qconfig", None)
            if qc is None:
                t.failed = True
                t.msg = "Returned model is missing qconfig."
                t.want = "qconfig is set (default QAT qconfig)"
                t.got = None
            else:
                # sanity: qconfig should define activation and weight fake-quant configs
                if not (hasattr(qc, "activation") and hasattr(qc, "weight") and qc.activation and qc.weight):
                    t.failed = True
                    t.msg = "qconfig must define activation and weight configs."
                    t.want = "qconfig.activation and qconfig.weight are set"
                    t.got = f"{qc}"
            cases.append(t)

            # --- 5) prepare_qat inserted observers/fake-quant modules ---
            # Expect at least one FakeQuantize present and Conv/Linear to have weight_fake_quant
            from torch.ao.quantization.fake_quantize import FakeQuantize
            fakeq_count = sum(isinstance(m, FakeQuantize) for m in qat_model.modules())

            t = test_case()
            if fakeq_count == 0:
                t.failed = True
                t.msg = "No FakeQuantize modules found; prepare_qat likely not applied."
                t.want = "≥ 1 FakeQuantize modules in the graph"
                t.got = "0"
            cases.append(t)

            t = test_case()
            any_weight_fq = False
            any_act_observer = False
            for m in qat_model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    if hasattr(m, "weight_fake_quant"):
                        any_weight_fq = True
                    if hasattr(m, "activation_post_process"):
                        any_act_observer = True
            if not any_weight_fq or not any_act_observer:
                t.failed = True
                t.msg = "Expected QAT observers/fake-quant on Conv/Linear weights and activations."
                t.want = "Modules have .weight_fake_quant and .activation_post_process"
                t.got = f"weight_fq={any_weight_fq}, act_obs={any_act_observer}"
            cases.append(t)

            # --- 6) Forward works on CPU in train() mode ---
            t = test_case()
            try:
                x = torch.randn(2, 3, 16, 16)  # CPU
                y = qat_model(x)  # train mode
                if not (isinstance(y, torch.Tensor) and y.shape[-1] == 3):
                    raise RuntimeError(f"Unexpected output: {type(y)} / shape={getattr(y, 'shape', None)}")
            except Exception as e:
                t.failed = True
                t.msg = f"QAT model forward pass failed: {e}"
                t.want = "Forward pass succeeds on CPU"
                t.got = "Exception raised"
            cases.append(t)

            # --- 7) Original model should remain unfused and have no QAT artifacts ---
            t = test_case()
            if not original_structure_ok(model):
                t.failed = True
                t.msg = "Original model structure was modified (fusion should not affect input)."
                t.want = "Original still has BN/ReLU in fusible positions"
                t.got = "Some positions changed"
            cases.append(t)

            t = test_case()
            had_qat_attrs = False
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    if hasattr(m, "weight_fake_quant") or hasattr(m, "activation_post_process"):
                        had_qat_attrs = True
                        break
            if had_qat_attrs:
                t.failed = True
                t.msg = "Original model contains QAT attributes; it must not be mutated."
                t.want = "No .weight_fake_quant / .activation_post_process on original"
                t.got = "Found QAT attributes on original"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error while executing tests: {e}"
            t.want = "Execution without errors"
            t.got = "Exception raised"
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
