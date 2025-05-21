import torch

def test_cuda():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
        # 测试 CUDA 计算
        x = torch.randn(2, 3).cuda()
        print("CUDA tensor:", x)
        print("设备:", x.device)
    else:
        print("CUDA 不可用，使用 CPU")
        print("可能的原因:")
        print("1. CUDA 版本不匹配")
        print("2. PyTorch 编译版本与系统不匹配")
        print("3. 环境变量配置问题")

test_cuda()