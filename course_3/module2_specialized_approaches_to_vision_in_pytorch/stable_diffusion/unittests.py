import copy
from types import FunctionType

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlai_grader.grading import print_feedback, test_case


def exercise_1(learner_func):
    def g():
        cases = []
        
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "cnn_feature_hierarchy has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        # ----- Mock Definitions ----------
        class MockBlock(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            
            def forward(self, x):
                return self.conv1(x)

        class MockResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, kernel_size=7, padding=3)
                
                # The learner code accesses model.layerX[0].conv1
                # Output channels increase at every stage to distinguish them
                self.layer1 = nn.Sequential(MockBlock(8, 16))   # Expected: 16 channels
                self.layer2 = nn.Sequential(MockBlock(16, 32))  # Expected: 32 channels
                self.layer3 = nn.Sequential(MockBlock(32, 64))  # Expected: 64 channels
                self.layer4 = nn.Sequential(MockBlock(64, 128)) # Expected: 128 channels
                
                self.fc = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                return x
        # ----- End Mock Definitions ----------

        # Setup Inputs
        sample_model = MockResNet()
        sample_model.eval()
        
        # Create a dummy input tensor (B, C, H, W)
        sample_img = torch.rand((1, 3, 224, 224), dtype=torch.float32)
        
        try:
            learner_sample = learner_func(sample_img, sample_model)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"cnn_feature_hierarchy raised an exception when called: {e}"
            t.want = "No exception"
            t.got = "Exception raised"
            return [t]

        # Return Type Check
        t = test_case()
        if not isinstance(learner_sample, dict):
            t.failed = True
            t.msg = "Incorrect return type from cnn_feature_hierarchy"
            t.want = dict
            t.got = type(learner_sample)
            return [t]

        # Keys Check
        expected_keys = {"conv1", "layer1", "layer2", "layer3", "layer4"}
        returned_keys = set(learner_sample.keys())
        missing_keys = expected_keys - returned_keys
        
        t = test_case()
        if missing_keys:
            t.failed = True
            t.msg = "cnn_feature_hierarchy return dictionary is missing required keys. Follow the exercise instructions to ensure all layers are captured."
            t.want = expected_keys
            t.got = returned_keys
            return [t]

        # Values Structure & Shape Check
        expected_channels = {
            "conv1": 8,
            "layer1": 16,
            "layer2": 32,
            "layer3": 64,
            "layer4": 128
        }

        for key in expected_keys:
            val = learner_sample[key]
            
            # Type Check
            t = test_case()
            if not isinstance(val, torch.Tensor):
                t.failed = True
                t.msg = f"Value for key '{key}' in returned dictionary is not a torch.Tensor"
                t.want = torch.Tensor
                t.got = type(val)
                return [t]
            
            # Dimension Check (ndim)
            t = test_case()
            if val.ndim != 4:
                t.failed = True
                t.msg = f"Tensor for key '{key}' has incorrect number of dimensions. Expected (B, C, H, W)."
                t.want = "4 dimensions"
                t.got = f"{val.ndim} dimensions (shape: {val.shape})"
                return [t]

            # Channel Check (Specific shape check to catch swapped layers)
            t = test_case()
            actual_channels = val.shape[1]
            target_channels = expected_channels[key]
            
            if actual_channels != target_channels:
                t.failed = True
                t.msg = (f"Incorrect channel size for key '{key}'. "
                         f"This usually means you registered the hook on the wrong layer. "
                         f"Check if you are mapping '{key}' to the correct model layer.")
                t.want = f"{target_channels} channels"
                t.got = f"{actual_channels} channels (shape: {val.shape})"
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []
        
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "feature_map_strip has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        # ----- Mock Definitions ----------
        class FixedOutput(nn.Module):
            def __init__(self, fixed_tensor):
                super().__init__()
                self.fixed_tensor = fixed_tensor
            def forward(self, x):
                return self.fixed_tensor

        class MockBlock(nn.Module):
            def __init__(self, in_c, out_c, size):
                super().__init__()
                # Create the predictable output tensor (Channel N has value N)
                self.fixed_out = torch.zeros(1, out_c, size, size)
                for c in range(out_c):
                    self.fixed_out[0, c, :, :] = float(c) 
                
                # Assign the FixedOutput layer to 'conv1'.
                self.conv1 = FixedOutput(self.fixed_out) 

            def forward(self, x):
                # Route execution through self.conv1 
                # so that the forward hook registered in Exercise 1 actually fires.
                return self.conv1(x)

        class MockResNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Structure matches ResNet-50 expectations for Exercise 1
                self.conv1 = MockBlock(3, 8, 112)                  # Output: 8 channels
                self.layer1 = nn.Sequential(MockBlock(8, 16, 56))  # Output: 16 channels
                self.layer2 = nn.Sequential(MockBlock(16, 32, 28)) # Output: 32 channels
                self.layer3 = nn.Sequential(MockBlock(32, 64, 14)) # Output: 64 channels
                self.layer4 = nn.Sequential(MockBlock(64, 128, 7)) # Output: 128 channels

            def forward(self, x):
                # Manually run the modules to trigger the hooks.
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                return x
        # ----- End Mock Definitions ----------

        try:
            sample_model = MockResNet()
            sample_model.eval()
            sample_img = torch.randn(1, 3, 224, 224) # Dummy input

            result = learner_func(sample_img, sample_model)
            
            # Type Check
            t = test_case()
            if not isinstance(result, list):
                t.failed = True
                t.msg = "feature_map_strip should return a list."
                t.want = list
                t.got = type(result)
                return [t]

            # Length Check
            t = test_case()
            if len(result) != 5:
                t.failed = True
                t.msg = f"Expected list of length 5 (conv1 + 4 layers), got {len(result)}"
                t.want = 5
                t.got = len(result)
                return [t]

            layer_names = ["conv1", "layer1", "layer2", "layer3", "layer4"]
            
            # Element Checks
            for idx, (val, name) in enumerate(zip(result, layer_names)):
                # Is Tensor?
                t = test_case()
                if not isinstance(val, torch.Tensor):
                    t.failed = True
                    t.msg = f"Element at index {idx} ({name}) is not a torch.Tensor"
                    t.want = torch.Tensor
                    t.got = type(val)
                    return [t]
                
                # Check 1: Dimensions (B, C, H, W)
                t = test_case()
                if val.ndim != 4:
                    t.failed = True
                    t.msg = f"Tensor for '{name}' must be 4-dimensional (B, C, H, W)."
                    t.want = 4
                    t.got = val.ndim
                    return [t]

                # Check 2: Batch and Channel sizes (1, 1, ...)
                t = test_case()
                if val.shape[0] != 1 or val.shape[1] != 1:
                    t.failed = True
                    t.msg = f"Tensor for '{name}' should have shape (1, 1, H, W) after selection."
                    t.want = "(1, 1, ...)"
                    t.got = f"({val.shape[0]}, {val.shape[1]}, ...)"
                    return [t]

                # Check 3: Upsampling Resolution (224x224)
                t = test_case()
                if val.shape[2:] != (224, 224):
                    t.failed = True
                    t.msg = f"Tensor for '{name}' was not upsampled correctly."
                    t.want = (224, 224)
                    t.got = val.shape[2:]
                cases.append(t)

                # Check 4: Normalization (0 to 1)
                t = test_case()
                # The Mock output implies values > 1.0 (channels 0 to N). 
                # If normalization logic is correct, min should be 0.0 and max 1.0.
                if val.min() < -1e-5 or val.max() > 1.0 + 1e-5:
                    t.failed = True
                    t.msg = f"Tensor for '{name}' is not normalized to [0, 1]."
                    t.want = "Values in range [0, 1]"
                    t.got = f"Range [{val.min():.4f}, {val.max():.4f}]"
                cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"feature_map_strip raised an exception: {e}. (Note: This might be due to an error in 'cnn_feature_hierarchy' if that function is incorrect)."
            t.want = "No exception"
            t.got = "Exception raised"
            return [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func):
    def g():
        cases = []

        # Sanity Check
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "saliency_map has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        # ----- Mock Definitions ----------
        class MockModel(nn.Module):
            def __init__(self, input_shape, num_classes=2):
                super().__init__()
                self.flat_dim = input_shape[0] * input_shape[1] * input_shape[2]
                # Linear layer: Output = Input * Weights + Bias
                # Therefore, d(Output)/d(Input) = Weights
                # Fix weights to random values to ensure non-zero gradients
                self.linear = nn.Linear(self.flat_dim, num_classes)
                
            def forward(self, x):
                # Flatten: (B, C, H, W) -> (B, C*H*W)
                out = x.view(x.size(0), -1)
                return self.linear(out)
        # ----- End Mock Definitions ----------

        try:
            H, W = 16, 12  # Use non-square dims to check H/W swapping bugs
            C = 3
            class_idx = 0
            
            # Setup Mock Model
            model = MockModel((C, H, W))
            model.eval()

            # Create dummy image (Batch=1, C=3, H=16, W=12)
            img_tensor = torch.randn(1, C, H, W)

            # Call learner function
            heatmap = learner_func(model, img_tensor, class_idx)

            
            # Type Check
            t = test_case()
            if not isinstance(heatmap, torch.Tensor):
                t.failed = True
                t.msg = "saliency_map should return a torch.Tensor"
                t.want = torch.Tensor
                t.got = type(heatmap)
                return [t]

            # Dimension Check (Should be 2D: H, W)
            t = test_case()
            if heatmap.dim() != 2:
                t.failed = True
                t.msg = f"saliency_map output must be 2D (H, W). Got {heatmap.dim()}D."
                t.want = "2D tensor"
                t.got = f"{heatmap.dim()}D shape {heatmap.shape}"
                return [t]
            
            # Shape Check
            t = test_case()
            if heatmap.shape != (H, W):
                t.failed = True
                t.msg = f"saliency_map output shape mismatch. Expected ({H}, {W})."
                t.want = f"({H}, {W})"
                t.got = f"{heatmap.shape}"
                return [t]

            # Gradient Detachment Check
            # The function should call .detach() before returning
            t = test_case()
            if heatmap.requires_grad:
                t.failed = True
                t.msg = "Returned heatmap tensor should be detached from computation graph."
                t.want = "requires_grad=False"
                t.got = "requires_grad=True"
                return [t]
            
            # Normalization Check (Range [0, 1])
            t = test_case()
            mn, mx = float(heatmap.min()), float(heatmap.max())
            if mn < -1e-5 or mx > 1.0 + 1e-5:
                t.failed = True
                t.msg = "Heatmap values must be normalized to range [0, 1]."
                t.want = "min >= 0, max <= 1"
                t.got = f"min={mn:.4f}, max={mx:.4f}"
                return [t]

            # Data Integrity Check
            t = test_case()
            if not torch.isfinite(heatmap).all():
                t.failed = True
                t.msg = "saliency_map contains NaN or infinite values. Check your normalization logic."
                t.want = "All finite values"
                t.got = "Contains NaNs/Infs"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error while executing saliency_map: {e}"
            t.want = "Execution without exception"
            t.got = "Exception raised"
            return [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func):
    def g():
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "simplified_cam has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        # Mock Definitions
        class FixedOutputLayer(nn.Module):
            def __init__(self, fixed_tensor):
                super().__init__()
                self.fixed_tensor = fixed_tensor
            def forward(self, x):
                return self.fixed_tensor

        class MockBlock(nn.Module):
            def __init__(self, channels, h, w):
                super().__init__()
                # Create predictable features for the hook to capture.
                # All ones means the CAM will just be the sum of weights.
                self.feats = torch.ones(1, channels, h, w)
                
                # The student hooks into 'conv3'
                self.conv3 = FixedOutputLayer(self.feats)

            def forward(self, x):
                # Route execution so the hook fires
                return self.conv3(x)

        class MockResNet(nn.Module):
            def __init__(self, channels, h, w, num_classes=2):
                super().__init__()
                # Define layer4 as a Sequential containing our MockBlock
                self.layer4 = nn.Sequential(
                    MockBlock(channels, h, w)
                )
                
                # Fully Connected layer for weights
                self.fc = nn.Linear(channels, num_classes)

            def forward(self, x):
                # Trigger the forward pass through the layers
                return self.layer4(x)
        
        try:
            # Params
            C_feat = 10
            H_feat, W_feat = 7, 7 # Small feature map
            H_img, W_img = 32, 28 # Input image size (non-square to check orientation)
            num_classes = 2
            
            # Setup Mock Model
            model = MockResNet(C_feat, H_feat, W_feat, num_classes)
            model.eval()
            
            # Dummy Image
            img = torch.randn(1, 3, H_img, W_img)

            # --- Test Case 1: Zero Weights ---
            # If FC weights are zero, the weighted sum (CAM) should be zero.
            with torch.no_grad():
                model.fc.weight.fill_(0.0)
                model.fc.bias.fill_(0.0)

            heat_zero = learner_func(model, img, class_idx=0)
            
            t = test_case()
            if not isinstance(heat_zero, torch.Tensor):
                t.failed = True
                t.msg = "simplified_cam should return a torch.Tensor"
                t.want = torch.Tensor
                t.got = type(heat_zero)
                return [t]
            
            t = test_case()
            if heat_zero.shape != (H_img, W_img):
                t.failed = True
                t.msg = f"CAM output shape mismatch. Expected ({H_img}, {W_img}). Check your upsampling logic."
                t.want = f"({H_img}, {W_img})"
                t.got = f"{heat_zero.shape}"
                return [t]

            t = test_case()
            # If weights are 0, CAM is 0. 
            # Normalization (0-min)/(max-min+eps) -> 0/eps = 0.
            if not torch.allclose(heat_zero, torch.zeros(H_img, W_img), atol=1e-5):
                t.failed = True
                t.msg = "Zero-weight case: CAM should be all zeros."
                t.want = "All zeros"
                t.got = f"Max value: {heat_zero.max().item()}"
                return [t]

            # --- Test Case 2: Standard Execution ---
            # Set weights to 1.0. 
            # Feats are 1.0. CAM = sum(1.0 * 1.0) = C_feat.
            # Normalization should result in valid range.
            with torch.no_grad():
                model.fc.weight.fill_(1.0)
            
            heat = learner_func(model, img, class_idx=0)

            # Dimensions Check
            t = test_case()
            if heat.dim() != 2:
                t.failed = True
                t.msg = f"CAM output must be 2D tensor. Got {heat.dim()}D."
                t.want = "2D tensor"
                t.got = f"{heat.shape}"
                return [t]

            # Range Check [0, 1]
            t = test_case()
            mn, mx = float(heat.min()), float(heat.max())
            if mn < -1e-5 or mx > 1.0 + 1e-5:
                t.failed = True
                t.msg = "CAM values must be normalized to range [0, 1]."
                t.want = "min >= 0, max <= 1"
                t.got = f"min={mn:.4f}, max={mx:.4f}"
                return [t]

            # Finite Check
            t = test_case()
            if not torch.isfinite(heat).all():
                t.failed = True
                t.msg = "CAM contains NaN or infinite values."
                t.want = "All finite values"
                t.got = "Contains NaNs/Infs"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error while executing simplified_cam: {e}"
            t.want = "Execution without exception"
            t.got = "Exception raised"
            return [t]

        return cases

    cases = g()
    print_feedback(cases)