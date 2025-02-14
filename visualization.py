import cv2
import os
from utils import normalize
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
def visualizer(pathes, anomaly_map, img_size, save_path, cls_name):
    for idx, path in enumerate(pathes):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        mask = normalize(anomaly_map[idx])
        vis = apply_ad_scoremap(vis, mask)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = os.path.join(save_path, 'imgs', cls_name[idx], cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename), vis)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Times New Roman 글꼴로 변경
plt.rc('font', family='Times New Roman')

def visualize_feature(image_level_features, pixel_level_features, image_labels, pixel_labels, legends, n_components=2, method='TSNE'):
    # TSNE 차원 축소 (perplexity와 learning_rate 조정)
    tsne = TSNE(n_components=n_components, random_state=5, perplexity=6, learning_rate=90)
    image_level_tsne = tsne.fit_transform(image_level_features)
    pixel_level_tsne = tsne.fit_transform(pixel_level_features)

    # 플롯 설정
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # fig.suptitle('Feature Visualization', fontsize=16)

    # 색상 정의
    color_map = {
        'Object-agnostic Normal': '#1f77b4',  # 파란색
        'Object-agnostic Abnormal': '#ff7f0e',  # 주황색
        'Global Normal': '#1f77b4',  # 파란색
        'Global Abnormal': '#ff7f0e',  # 주황색
        'Visual Normal': '#2ca02c',  # 초록색
        'Visual Abnormal': '#d62728',  # 빨간색
        'Local Normal': '#9467bd',  # 보라색
        'Local Abnormal': '#8c564b'  # 갈색
    }
    anoamlyclip_text_prompts = ['Object-agnostic Normal', 'Object-agnostic Abnormal']
    global_text_prompts = ['Global Normal', 'Global Abnormal']
    local_text_prompts = ['Local Normal', 'Local Abnormal']
    text_prompts = global_text_prompts + local_text_prompts
    # 이미지 레벨 특징 플롯
    for label in np.unique(image_labels):
        mask = np.array(image_labels) == label
        color = color_map[label]
        if label in global_text_prompts:
            marker = '*'
        elif label in local_text_prompts:
            marker = 'P'
        elif label in anoamlyclip_text_prompts:
            marker = 'X'
        else:
            marker = 'o'

        size = 200 if label in text_prompts else 50
        
        ax1.scatter(image_level_tsne[mask, 0], image_level_tsne[mask, 1], 
                    c=color, label=label, marker=marker, s=size, alpha=0.5)
    ax1.grid(True)
    ax1.set_title('Image-level features', fontsize=16)

    ax1.legend(loc='lower right')  # 범례를 박스 안의 오른쪽 하단에 배치

    # 픽셀 레벨 특징 플롯
    for label in np.unique(image_labels):
        mask = np.array(image_labels) == label
        color = color_map[label]
        if label in global_text_prompts:
            marker = '*'
        elif label in local_text_prompts:
            marker = 'P'
        elif label in anoamlyclip_text_prompts:
            marker = 'X'
        else:
            marker = 'o'

        size = 200 if label in text_prompts else 50
        
        ax2.scatter(pixel_level_tsne[mask, 0], pixel_level_tsne[mask, 1], 
                    c=color, label=label, marker=marker, s=size, alpha=0.5)
    ax2.grid(True)
    ax2.set_title('Pixel-level features', fontsize=16)

    # 모든 글씨 폰트 크기를 논문에 맞춰 Times New Roman으로 변경
    plt.tight_layout()
    plt.savefig('feature_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_similarity_scores(similarity_scores, save_dir="similarity_plots", bin_count=40, dataset_name=None):
    """
    Plots similarity scores for normal and anomaly cases across multiple object classes,
    saving each plot as an individual image.
    
    Args:
    similarity_scores (dict): Dictionary containing similarity scores for different object classes.
                              Keys should be object class names, and values should be another dictionary
                              with keys 'normal' and 'anomaly' containing lists of similarity scores.
    save_dir (str): Directory where individual plots will be saved.
    bin_count (int): Number of bins to use for the histograms.
    """
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define global bins based on fixed range [0, 1]
    bins = np.linspace(0, 1, bin_count + 1)  # Create bin edges from 0 to 1 with 'bin_count' bins

    for obj_class, scores in similarity_scores.items():
        # Create a new figure for each object class
        fig, ax = plt.subplots(figsize=(6, 4))

        # Add a light gray background to the plot
        ax.set_facecolor('#f2f2f2')
        
        # Extract normal and anomaly scores
        normal_scores = scores['normal']
        anomaly_scores = scores['anomaly']
        
        # Plot histograms for normal and anomaly similarity scores with fixed bins and borders
        ax.hist(normal_scores, bins=bins, alpha=0.6, label='Normal', color='steelblue', edgecolor='black', linewidth=1.2)
        ax.hist(anomaly_scores, bins=bins, alpha=0.6, label='Anomaly', color='coral', edgecolor='black', linewidth=1.2)
        
        # Set x-axis ticks to 0, 0.5, and 1, and adjust font size
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.tick_params(axis='x', labelsize=22)  # Increase x-axis font size here
        
        # Set the x-axis limits
        ax.set_xlim(0, 1)
        
        # Hide y-axis labels
        ax.set_yticks([])

        # Draw a vertical line at x=0.5
        ax.axvline(x=0.5, color='white', linestyle='--', linewidth=2)

        # Display the legend with a larger font size
        ax.legend(fontsize=16)  # Adjust the legend font size

        # Set the title as the object class name with a larger font size
        if dataset_name:
            ax.set_title(f"{obj_class.capitalize()}", fontsize=22, weight='bold')
        else:
            ax.set_title(f"{obj_class.capitalize()}", fontsize=22, weight='bold')

        # Add a subtle grid with dashed lines
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure for the current object class
        plot_path = os.path.join(save_dir, f"{obj_class.capitalize()}_similarity_scores.png")
        plt.savefig(plot_path, dpi=300)  # Save at high resolution
        plt.close()

    print(f"Plots saved in {save_dir}")
