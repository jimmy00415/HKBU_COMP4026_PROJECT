"""Quick training pipeline test — 3 epochs, CPU."""
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')

from scripts.train_expression import train_expression

result = train_expression(
    dataset='fer2013',
    data_root='data/fer2013',
    backbone='resnet18',
    num_classes=7,
    pretrained=True,
    epochs=3,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    patience=7,
    device='cpu',
    seed=42,
    output_dir='results/expression_training',
    checkpoint_dir='checkpoints',
    use_tensorboard=False,
    augmentation=True,
)
print(f"Best val acc: {result['best_val_acc']:.4f}")
print(f"Checkpoint: {result['checkpoint_path']}")
