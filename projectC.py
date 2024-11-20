import os
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
from scipy.cluster.hierarchy import ClusterWarning
from scipy.cluster.hierarchy import fcluster
import shutil
from sklearn.manifold import TSNE

# 数据根目录
data_root = 'C:/Users/86198/Desktop/智能媒体工作坊/20关/CSV'
level_ids = [f'Level{i + 1}' for i in range(20)]  # 假设关卡编号从1到20


# 提取每关特征（总长度和平均曲率）的函数
def extract_level_features(level_dir):
    total_length = 0
    total_curvature = 0
    num_players = 0

    for filename in os.listdir(level_dir):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(level_dir, filename))
            points = df[['x', 'y']].values

            # 计算当前文件的路径总长度
            length = np.sum(np.sqrt(np.diff(points, axis=0) ** 2).sum(axis=1))
            total_length += length

            # 计算当前文件的曲率，简化为角度变化
            curvature = compute_curvature(points)
            total_curvature += curvature

            num_players += 1

    if num_players > 0:
        avg_length = total_length / num_players
        avg_curvature = total_curvature / num_players
        return avg_length, avg_curvature
    else:
        return 0, 0

def compute_curvature(points):

    if len(points) < 3:
        return 0

    total_curvature = 0
    for i in range(1, len(points) - 1):
        p0, p1, p2 = points[i - 1], points[i], points[i + 1]
        dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
        dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]

        # 计算斜率
        k1, k2 = dy1 / dx1 if dx1 != 0 else float('inf'), dy2 / dx2 if dx2 != 0 else float('inf')

        # 斜率存在且不是无穷时，计算夹角
        if np.isfinite(k1) and np.isfinite(k2):
            theta = np.abs(np.arctan(k2) - np.arctan(k1))
            # 转换为弧度制下的曲率
            curvature = np.abs(2 * np.sin(theta / 2) / np.linalg.norm(np.array([dx2, dy2])))
            total_curvature += curvature

    return total_curvature / (len(points) - 2)



# 提取所有关卡的特征并打印
level_features = {level_id: extract_level_features(os.path.join(data_root, level_id)) for level_id in level_ids}
print(level_features)


def normalize_features_dict(level_features):
    # 分别提取所有级别的总长度和平均曲率
    all_lengths = np.array([features[0] for features in level_features.values()])
    all_curvatures = np.array([features[1] for features in level_features.values()])

    # 对总长度和平均曲率分别进行归一化
    normalized_lengths = (all_lengths - all_lengths.min()) / (all_lengths.max() - all_lengths.min())
    normalized_curvatures = (all_curvatures - all_curvatures.min()) / (all_curvatures.max() - all_curvatures.min())

    # 创建归一化后的特征字典
    normalized_features = {level_id: (normalized_lengths[idx], normalized_curvatures[idx])
                                for idx, level_id in enumerate(level_features.keys())}

    return normalized_features

# 调用归一化函数
normalized_level_features = normalize_features_dict(level_features)
# 归一化数据
print(normalized_level_features)

# 初始化差异度矩阵
distance_matrix = np.zeros((20, 20))

# 计算差异度（这里使用欧氏距离作为示例）
for i in range(20):
    for j in range(i + 1, 20):
        feat_i = normalized_level_features[level_ids[i]]
        feat_j = normalized_level_features[level_ids[j]]
        diff = distance.euclidean(feat_i, feat_j)
        distance_matrix[i, j] = diff
        distance_matrix[j, i] = diff  # 由于是对称矩阵，也设置下半部分

# 使用热力图可视化差异度矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix, annot=True, cmap='coolwarm', xticklabels=level_ids, yticklabels=level_ids)
plt.title('Distance Matrix between Levels')
plt.xlabel('Level')
plt.ylabel('Level')
plt.show()

#df = pd.DataFrame(distance_matrix, columns=['Level1','Level2','Level3','Level4','Level5','Level6','Level7','Level8','Level9','Level10','Level11','Level12','Level13','Level14','Level15','Level16','Level17','Level18','Level19','Level20'])
#df.to_excel('distance_matrix.xlsx', index=False)
df = pd.DataFrame(distance_matrix, columns=['Level' + str(i) for i in range(1, 21)])
df.insert(0, 'Level', ['Level' + str(i) for i in range(1, 21)])
df.index = range(len(df)) 
df.to_excel('distance_matrix.xlsx', index=False, engine='openpyxl')

warnings.filterwarnings("ignore", category=ClusterWarning)

# 层次聚类
linked = linkage(distance_matrix, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=range(1, 21),
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('level')
plt.ylabel('distance')
plt.show()


max_d = 1.5  #选择的阈值
clusters = fcluster(linked, max_d, criterion='distance')

# 打印簇分配结果
print("Cluster assignments:", clusters)

features = np.array(list(normalized_level_features.values()))
levels = np.array(list(normalized_level_features.keys()))

colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'black'] # 根据你的簇数量调整颜色列表

# 绘制散点图
plt.figure(figsize=(10, 6))
for label in np.unique(clusters):
    # 选择属于当前簇的数据点
    cluster_points = features[clusters == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=colors[label % len(colors)],
                label=f'Cluster {label}')
# 添加图例和标题
plt.legend()
plt.title('Levels Clustered by Hierarchical Clustering')
plt.xlabel('Total Length')
plt.ylabel('Average Curvature')
plt.grid(True)
plt.show()



png_folder = 'C:/Users/86198/Desktop/智能媒体工作坊/20关/PNG'
# 创建一个新的文件夹来存放聚类后的图片
clustered_images_folder = 'C:/Users/86198/Desktop/智能媒体工作坊/20关/clustered_images'
if not os.path.exists(clustered_images_folder):
    os.makedirs(clustered_images_folder)

# 遍历每个关卡（这里用索引表示）和它的簇编号
for i, cluster_id in enumerate(clusters):
    # 注意：簇编号可能需要+1来匹配可能的从1开始的簇编号习惯
    cluster_id_for_folder = cluster_id
    # 构建源图片路径
    source_img_path = os.path.join(png_folder, f'Level{i + 1}.png')  # 假设图片命名是 level_1.png, level_2.png, ...

    # 构建目标文件夹路径
    target_folder_path = os.path.join(clustered_images_folder, f'cluster_{cluster_id_for_folder}')

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
        # 构建目标图片路径
    target_img_path = os.path.join(target_folder_path, os.path.basename(source_img_path))

    # 将图片复制到目标文件夹
    shutil.copy2(source_img_path, target_img_path)
print("图片已按簇分类完毕。")

tsne = TSNE(n_components=2, perplexity=5, verbose=1)
embedded_coords = tsne.fit_transform(distance_matrix)

# 可视化结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(embedded_coords[:, 0], embedded_coords[:, 1])
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE visualization of game levels')
# 如果你想为每个关卡添加标签，你可以这样做（但这里我们没有实际的关卡名称）
for i, txt in enumerate(level_ids):
    plt.annotate(txt, (embedded_coords[i, 0], embedded_coords[i, 1]))
plt.grid(True)
plt.show()
