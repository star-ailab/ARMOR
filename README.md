# ARMOR
ARMOR's Implementation (under review at SIAM's SIMODS)

# Usage
- For benchmark experiments on CIFAR-10 (Section 3 in the paper), run:
  - python armor_udr_cifar10.py for generalization of UDR with our $ARMOR_D$ method
  - python armor_mart_cifar10.py for generalization of MART with our $ARMOR_D$ method
  - python armor_trades_cifar10.py for generalization of TRADES with our $ARMOR_D$ method
- For additional experiments on different variants of $ARMOR_D$ on MNIST dataset (Supplementary Material), run:
  - python armor_f_mnist.py for $ARMOR_{\alpha}$ ($adv_s$)
  - python armor_f_worig_mnist.py for $ARMOR_{\alpha}$ ($adv_s+nat$)
  - python armor_KL_mnist.py for $ARMOR_{KL}$ ($adv_s$)
  - python armor_KL_worig_mnist.py for $ARMOR_{KL}$ ($adv_s+nat$)

# Acknowldegements
For data preprocessing and common training operations, the code leverages the following prior works:
- https://github.com/tuananhbui89/Unified-Distributional-Robustness (ICLR 2022)
- https://github.com/YisenWang/MART (ICLR2020)


