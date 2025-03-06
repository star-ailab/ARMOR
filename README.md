# ARMOR
$ARMOR_D$ Implementation (under review at SIAM's SIMODS)

# Usage Examples
- For benchmark experiments on CIFAR-100, you can run:
  - ```bash python armor_udr_cifar100_KL_bisec.py''' for generalization of UDR with our $ARMOR_D$ method
  - python armor_mart_cifar100_KL_bisec.py for generalization of MART with our $ARMOR_D$ method
  - python armor_trades_cifar100_KL_bisec.py for generalization of TRADES with our $ARMOR_D$ method
- For benchmark experiments on CIFAR-10, you may run:
  - python armor_udr_cifar10.py for generalization of UDR with our $ARMOR_D$ method
  - python armor_mart_cifar10.py for generalization of MART with our $ARMOR_D$ method
  - python armor_trades_cifar10.py for generalization of TRADES with our $ARMOR_D$ method

- *Remark*: For UDR methods, after training is done, you can evaluate the robustified model by running:
  - python eval_cifar100 and python eval_cifar10
    
- For additional experiments without adversarial training (i.e., no defense mechanism), you can run:
  - python ce_cifar10.py and python ce_cifar100.py
    
- For additional experiments on different variants of $ARMOR_D$ on with alpha, Renyi, and KL divergences on MNIST dataset, you can run:
  - $KL-$ Divergence
    - python armor_KL_mnist.py for $ARMOR_{KL}$ ($adv_s$)
    - python armor_KL_worig_mnist.py for $ARMOR_{KL}$ ($adv_s+nat$)
  - $\alpha-$ Divergence
    - python armor_f_mnist.py for $ARMOR_{\alpha}$ ($adv_s$)
    - python armor_f_worig_mnist.py for $ARMOR_{\alpha}$ ($adv_s+nat$)
  - $R\'{e}nyi-$ Divergence:
    - python armor_Renyi_worig_mnist.py for $ARMOR_{R\'{e}n}$ ($adv_s+nat$)


# Requirements
- Python 3.8.0
- PyTorch 1.13.1
- For data preprocessing and common training operations, parts of the code is based on the following GitHub repositories:
  - UDR: https://github.com/tuananhbui89/Unified-Distributional-Robustness (ICLR)
  - MART: https://github.com/YisenWang/MART (ICLR)
  - TRADES: https://github.com/yaodongyu/TRADES (ICML)


