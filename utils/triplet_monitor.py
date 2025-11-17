"""
Real-time Triplet Loss Training Monitor

Use this to monitor your actual model training with live visualizations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import deque
import time

class TripletTrainingMonitor:
    """
    Monitor triplet loss training in real-time

    Usage:
        monitor = TripletTrainingMonitor(window_size=100)

        # During training loop:
        for batch in dataloader:
            ...
            loss, metrics = loss_fn(anchors, positives, negatives)
            monitor.update(metrics, step=global_step)

            if step % 50 == 0:
                monitor.plot()  # Show current state

        # After training:
        monitor.save_report('training_report.html')
    """

    def __init__(self, window_size=100, margin=0.5):
        self.window_size = window_size
        self.margin = margin

        # Use deques for efficient rolling window
        self.loss_history = deque(maxlen=window_size)
        self.pos_dist_history = deque(maxlen=window_size)
        self.neg_dist_history = deque(maxlen=window_size)
        self.separation_history = deque(maxlen=window_size)
        self.violations_history = deque(maxlen=window_size)
        self.steps = deque(maxlen=window_size)

        # Full history for final report
        self.full_history = {
            'loss': [],
            'pos_dist': [],
            'neg_dist': [],
            'separation': [],
            'violations': [],
            'steps': [],
            'timestamps': []
        }

        self.start_time = time.time()

        # Setup matplotlib for live plotting
        plt.ion()
        self.fig = None
        self.axes = None

    def update(self, metrics, step):
        """Update monitor with new metrics from a training step"""

        # Extract metrics
        loss = metrics.get('loss', 0)
        pos_dist = metrics.get('pos_dist_mean', 0)
        neg_dist = metrics.get('neg_dist_mean', 0)
        separation = neg_dist - pos_dist
        violations = metrics.get('margin_violations_pct', 0)

        # Update rolling window
        self.loss_history.append(loss)
        self.pos_dist_history.append(pos_dist)
        self.neg_dist_history.append(neg_dist)
        self.separation_history.append(separation)
        self.violations_history.append(violations)
        self.steps.append(step)

        # Update full history
        self.full_history['loss'].append(loss)
        self.full_history['pos_dist'].append(pos_dist)
        self.full_history['neg_dist'].append(neg_dist)
        self.full_history['separation'].append(separation)
        self.full_history['violations'].append(violations)
        self.full_history['steps'].append(step)
        self.full_history['timestamps'].append(time.time() - self.start_time)

    def plot(self, save_path=None):
        """Plot current training state"""

        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
            self.fig.suptitle('Triplet Loss Training Monitor', fontsize=16, fontweight='bold')

        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()

        steps = list(self.steps)

        # 1. Loss
        self.axes[0, 0].plot(steps, list(self.loss_history), 'b-', linewidth=2)
        self.axes[0, 0].set_title('Loss', fontweight='bold')
        self.axes[0, 0].set_xlabel('Step')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)

        # 2. Distances
        self.axes[0, 1].plot(steps, list(self.pos_dist_history), 'g-', label='Pos', linewidth=2)
        self.axes[0, 1].plot(steps, list(self.neg_dist_history), 'r-', label='Neg', linewidth=2)
        self.axes[0, 1].set_title('Distances (Pos vs Neg)', fontweight='bold')
        self.axes[0, 1].set_xlabel('Step')
        self.axes[0, 1].set_ylabel('Distance')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)

        # 3. Separation
        self.axes[1, 0].plot(steps, list(self.separation_history), 'purple', linewidth=2)
        self.axes[1, 0].axhline(y=self.margin, color='r', linestyle='--', label=f'Margin ({self.margin})')
        self.axes[1, 0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        self.axes[1, 0].set_title('Separation (Neg - Pos)', fontweight='bold')
        self.axes[1, 0].set_xlabel('Step')
        self.axes[1, 0].set_ylabel('Separation')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True, alpha=0.3)

        # 4. Violations
        self.axes[1, 1].plot(steps, list(self.violations_history), 'orange', linewidth=2)
        self.axes[1, 1].set_title('Margin Violations', fontweight='bold')
        self.axes[1, 1].set_xlabel('Step')
        self.axes[1, 1].set_ylabel('Violations (%)')
        self.axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.pause(0.001)  # Force update

    def get_summary(self):
        """Get current training summary"""
        if not self.full_history['loss']:
            return "No training data yet"

        current_loss = self.full_history['loss'][-1]
        current_pos = self.full_history['pos_dist'][-1]
        current_neg = self.full_history['neg_dist'][-1]
        current_sep = self.full_history['separation'][-1]
        current_viol = self.full_history['violations'][-1]

        # Calculate improvements
        initial_loss = self.full_history['loss'][0]
        loss_reduction = (initial_loss - current_loss) / initial_loss * 100

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              TRIPLET TRAINING SUMMARY                        ║
╚══════════════════════════════════════════════════════════════╝

📊 Current Metrics (Step {self.full_history['steps'][-1]}):
  • Loss:            {current_loss:.4f} (↓ {loss_reduction:.1f}% from start)
  • Pos Distance:    {current_pos:.4f}
  • Neg Distance:    {current_neg:.4f}
  • Separation:      {current_sep:.4f} (target: > {self.margin})
  • Violations:      {current_viol:.1f}%

📈 Best Achieved:
  • Lowest Loss:     {min(self.full_history['loss']):.4f}
  • Best Separation: {max(self.full_history['separation']):.4f}
  • Lowest Violations: {min(self.full_history['violations']):.1f}%

⏱️  Training Time: {self.full_history['timestamps'][-1]:.1f}s

✅ Health Check:
  • Loss decreasing:     {'✓' if current_loss < initial_loss else '✗'}
  • Separation > margin: {'✓' if current_sep > self.margin else '✗'}
  • Violations < 20%:    {'✓' if current_viol < 20 else '✗'}
"""
        return summary

    def save_report(self, output_path='training_report.png'):
        """Save final training report"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Triplet Loss Training Report', fontsize=18, fontweight='bold')

        steps = self.full_history['steps']

        # 1. Loss over time
        axes[0, 0].plot(steps, self.full_history['loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('Loss Evolution', fontweight='bold')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Distances
        axes[0, 1].plot(steps, self.full_history['pos_dist'], 'g-', label='Positive', linewidth=2)
        axes[0, 1].plot(steps, self.full_history['neg_dist'], 'r-', label='Negative', linewidth=2)
        axes[0, 1].set_title('Distance Evolution', fontweight='bold')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Distance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Separation
        axes[0, 2].plot(steps, self.full_history['separation'], 'purple', linewidth=2)
        axes[0, 2].axhline(y=self.margin, color='r', linestyle='--', label=f'Margin')
        axes[0, 2].fill_between(steps, self.margin, max(self.full_history['separation']),
                               alpha=0.2, color='green', label='Safe zone')
        axes[0, 2].set_title('Separation (Neg - Pos)', fontweight='bold')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Separation')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Violations
        axes[1, 0].plot(steps, self.full_history['violations'], 'orange', linewidth=2)
        axes[1, 0].axhline(y=10, color='green', linestyle='--', label='Target (<10%)')
        axes[1, 0].set_title('Margin Violations', fontweight='bold')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Violations (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Training speed (steps/sec)
        if len(self.full_history['timestamps']) > 1:
            time_diffs = np.diff(self.full_history['timestamps'])
            steps_per_sec = 1 / (time_diffs + 1e-8)
            axes[1, 1].plot(steps[1:], steps_per_sec, 'cyan', linewidth=2)
            axes[1, 1].set_title('Training Speed', fontweight='bold')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Steps/Second')
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""
Final Statistics:

Loss: {self.full_history['loss'][-1]:.4f}
  (↓ {(self.full_history['loss'][0] - self.full_history['loss'][-1]):.4f})

Pos Distance: {self.full_history['pos_dist'][-1]:.4f}
Neg Distance: {self.full_history['neg_dist'][-1]:.4f}

Separation: {self.full_history['separation'][-1]:.4f}
  (Target: > {self.margin})

Violations: {self.full_history['violations'][-1]:.1f}%

Total Steps: {len(steps)}
Training Time: {self.full_history['timestamps'][-1]:.1f}s
        """
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Report saved to {output_path}")

        return fig


def compute_triplet_metrics(anchor, positive, negative, margin=0.5, distance='cosine'):
    """
    Compute detailed triplet metrics

    Returns dict with:
        - loss
        - pos_dist_mean, pos_dist_std
        - neg_dist_mean, neg_dist_std
        - margin_violations, margin_violations_pct
        - avg_separation
        - hard_triplets
    """

    if distance == 'cosine':
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)
        pos_dist = 1 - pos_sim
        neg_dist = 1 - neg_sim
    else:  # euclidean
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

    # Triplet loss
    losses = F.relu(pos_dist - neg_dist + margin)
    loss = losses.mean()

    # Metrics
    metrics = {
        'loss': loss.item(),
        'pos_dist_mean': pos_dist.mean().item(),
        'pos_dist_std': pos_dist.std().item(),
        'neg_dist_mean': neg_dist.mean().item(),
        'neg_dist_std': neg_dist.std().item(),
        'margin_violations': (losses > 0).sum().item(),
        'margin_violations_pct': (losses > 0).float().mean().item() * 100,
        'avg_separation': (neg_dist - pos_dist).mean().item(),
        'hard_triplets': (losses > margin * 0.5).sum().item(),
    }

    return metrics


if __name__ == "__main__":
    # Demo: simulate training
    print("Running demo simulation...")

    monitor = TripletTrainingMonitor(window_size=200, margin=0.5)

    for step in range(300):
        # Simulate improving metrics
        progress = step / 300

        # Create fake metrics
        metrics = {
            'loss': 0.5 * (1 - progress * 0.8) + np.random.rand() * 0.1,
            'pos_dist_mean': 0.3 * (1 - progress * 0.6) + np.random.rand() * 0.05,
            'neg_dist_mean': 0.6 + progress * 0.3 + np.random.rand() * 0.05,
            'margin_violations_pct': 50 * (1 - progress) + np.random.rand() * 10,
        }

        monitor.update(metrics, step)

        if step % 50 == 0:
            print(f"\nStep {step}:")
            print(monitor.get_summary())
            monitor.plot()

    # Save final report
    monitor.save_report('demo_training_report.png')
    print("\n✓ Demo complete! Check demo_training_report.png")
