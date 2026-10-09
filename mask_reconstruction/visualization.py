import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def mask_overlay(rgb, masks, alpha=0.5):
    """RGB 画像にインスタンスごとに色分けしたマスクを重ねる．"""
    overlay = rgb.copy()
    cmap = plt.get_cmap("hsv", len(masks) + 1)
    for i, m in enumerate(masks):
        overlay[m] = overlay[m] * (1 - alpha) + np.array(cmap(i)[:3]) * 255 * alpha
    return overlay.astype(np.uint8)


def save_comparison(sample, results, out_path):
    """原画像・GT・各手法のマスクを並べた比較図を保存する．

    results: {手法名: マスクのリスト}
    """
    rgb = cv2.cvtColor(sample.rgb, cv2.COLOR_BGR2RGB)
    panels = [("Original RGB", rgb), ("Ground Truth", mask_overlay(rgb, sample.gt_masks))]
    panels += [(name, mask_overlay(rgb, m)) for name, m in results.items()]

    cols = 3
    rows = (len(panels) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, (title, img) in zip(np.ravel(axs), panels):
        ax.imshow(img)
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
