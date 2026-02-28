import pyperf
import torch

CASES = [
    ("tiny", 1, 4, 8, 8, 3, 1, 1),
    ("small", 3, 8, 32, 32, 3, 1, 1),
    ("medium", 3, 16, 64, 64, 5, 1, 2),
]


def build_case(case):
    _, ic, oc, h, w, k, stride, pad = case
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(
        ic,
        oc,
        k,
        stride=stride,
        padding=pad,
        bias=True,
        dtype=torch.float64,
        device="cpu",
    )
    conv.eval()
    with torch.no_grad():
        conv.bias.zero_()
    x = torch.randn((1, ic, h, w), dtype=torch.float64, device="cpu")
    return conv, x


def bench_case(case):
    conv, x = build_case(case)
    conv(x)

    def run():
        conv(x)

    return run


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.mkldnn.enabled = True

    runner = pyperf.Runner()
    runner.argparser.add_argument(
        "--case",
        default="all",
        help="Case name to run or 'all'.",
    )
    args = runner.parse_args()
    runner.metadata["torch_dtype"] = "float64"
    runner.metadata["torch_threads"] = 1
    runner.metadata["torch_mkldnn"] = torch.backends.mkldnn.enabled
    cases = CASES if args.case == "all" else [c for c in CASES if c[0] == args.case]
    if not cases:
        names = ", ".join(c[0] for c in CASES)
        raise SystemExit(f"Unknown case '{args.case}'. Valid cases: {names}")

    for case in cases:
        name = case[0]
        runner.bench_func(f"torch_conv/{name}", bench_case(case))


if __name__ == "__main__":
    main()
