import torch


_CASES = [
    ("tiny", 1, 4, 8, 8, 3, 1, 1),
    ("small", 3, 8, 32, 32, 3, 1, 1),
    ("medium", 3, 16, 64, 64, 5, 1, 2),
]


class TimeTorchConv2d:
    params = _CASES
    param_names = ["case"]

    def setup(self, case):
        _, ic, oc, h, w, k, stride, pad = case
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        torch.manual_seed(0)

        self.conv = torch.nn.Conv2d(
            ic,
            oc,
            k,
            stride=stride,
            padding=pad,
            bias=True,
            dtype=torch.float64,
            device="cpu",
        )
        self.conv.eval()
        with torch.no_grad():
            self.conv.bias.zero_()
        self.x = torch.randn((1, ic, h, w), dtype=torch.float64, device="cpu")

        # Warm up once to initialize kernels.
        self.conv(self.x)

    def teardown(self, case):
        torch.set_grad_enabled(True)

    def time_forward(self, case):
        self.conv(self.x)
