import torch
import torch.nn as nn
import numpy as np
import os

# ==================== 配置参数 ====================
CONFIG = {
    'feature_input_txt': 'feature_input.txt',   # 待预测文件名
    'start_column': 0,                          # 待预测文件40维输入的起始列，从0起算
    'cm1_dir': 'CM1',                      # 预训练模型cm1路径（用于计算准确自由能）
    'cm2_dir': 'CM2',                         # 分类模型cm2路径（用于分类预测）
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'category_weights': [1, 1, 2],      # 三个类别的权重：类别1:类别2:类别3 = 1:1:2
}

# ==================== 预训练回归模型 ====================
def build_pretrain_model(n_in):
    model = nn.Sequential(
        nn.Linear(n_in, 256), nn.ReLU(),
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 1)
    )
    return model

# ==================== 分类融合模型 ====================
class FusionModel(nn.Module):
    def __init__(self, n_in, device):
        super().__init__()
        self.device = device

        self.backbone1 = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(41, 512), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        out1 = self.backbone1(x)
        fused = torch.cat([out1, x], dim=1)
        cls_out = self.classifier(fused)
        return cls_out

# ==================== 读取待预测数据 ====================
def load_predict_data(txt_path):
    features = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            feat = list(map(float, cols[CONFIG['start_column']:CONFIG['start_column']+40]))
            features.append(feat)
    return np.array(features, dtype=np.float32)

# ==================== 加载所有模型 ====================
def load_all_models():
    device = CONFIG['device']
    pretrain_model_list = []
    cls_model_list = []

    for fold_idx in range(5):
        # 1. 加载预训练回归模型（用于计算自由能）
        pretrain_path = os.path.join(CONFIG['cm1_dir'], f'fold_{fold_idx}', 'best_model.pth')
        pretrain_model = build_pretrain_model(n_in=40).to(device)
        pretrain_model.load_state_dict(torch.load(pretrain_path, map_location=device))
        pretrain_model.eval()
        pretrain_model_list.append(pretrain_model)

        # 2. 加载分类模型（用于分类预测）
        model_path = os.path.join(CONFIG['cm2_dir'], f'fold_{fold_idx}', 'best_model.pth')
        cls_model = FusionModel(n_in=40, device=device).to(device)
        cls_model.load_state_dict(torch.load(model_path, map_location=device))
        cls_model.eval()
        cls_model_list.append(cls_model)

    print("5折预训练模型与分类模型全部加载完成")
    return pretrain_model_list, cls_model_list

# ==================== 集成推理 ====================
def predict_with_5folds(pretrain_model_list, cls_model_list, features):
    device = CONFIG['device']
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    all_energies = []  # 来自正确预训练模型的溶解自由能
    all_probs = []     # 来自分类模型的三分类概率

    with torch.no_grad():
        for p_model, c_model in zip(pretrain_model_list, cls_model_list):
            # 计算自由能（用正确的预训练权重）
            energies = p_model(features_tensor)
            all_energies.append(energies.flatten().cpu().numpy())
            
            # 计算分类概率
            outputs = c_model(features_tensor)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_energies = np.array(all_energies)  # shape: (5, N)
    all_probs = np.array(all_probs)        # shape: (5, N, 3)

    # 溶解自由能平均值
    avg_energies = np.mean(all_energies, axis=0)

    # soft voting：5折模型等权重平均概率
    soft_vote_probs = np.mean(all_probs, axis=0)

    # 1:1:2 类别加权软投票
    category_weights = np.array(CONFIG['category_weights'], dtype=np.float32)
    weighted_vote_probs = soft_vote_probs * category_weights

    # 最终结果：基于加权概率判定，类别编号为1/2/3
    final_preds = np.argmax(weighted_vote_probs, axis=1) + 1

    return all_energies, all_probs, avg_energies, soft_vote_probs, weighted_vote_probs, final_preds

# ==================== 主推理函数 ====================
def main():
    # 1. 加载数据
    print(f"加载待预测数据：{CONFIG['feature_input_txt']}")
    features = load_predict_data(CONFIG['feature_input_txt'])
    print(f"数据加载完成，共 {len(features)} 个样本，40维特征\n")

    # 2. 加载模型
    pretrain_model_list, cls_model_list = load_all_models()

    # 3. 集成推理
    all_energies, all_probs, avg_energies, soft_vote_probs, weighted_vote_probs, final_preds = \
        predict_with_5folds(pretrain_model_list, cls_model_list, features)

    # 4. 按指定格式输出
    print("\n" + "="*60)
    print("预测结果")
    print("="*60 + "\n")
    
    classification_results={1:"soluble", 2: "partly soluble", 3:"insoluble"}
    
    for idx in range(len(features)):
        print(f"@task {idx+1}")
        print("Free Energy Result: ")
        e1, e2, e3, e4, e5 = all_energies[:, idx]
        print(f"model 1-5:     {e1:>8.4f} {e2:>8.4f} {e3:>8.4f} {e4:>8.4f} {e5:>8.4f}")
        print(f"average:      {avg_energies[idx]:>8.4f}")
        print()
        print("Dissolution Classification")
        for fold_idx in range(5):
            p1, p2, p3 = all_probs[fold_idx, idx]
            print(f"model {fold_idx+1}: {p1:>8.4f} {p2:>8.4f} {p3:>8.4f}")
        print()
        s1, s2, s3 = soft_vote_probs[idx]
        print(f"{'soft voting':>26} :  {s1:>8.4f} {s2:>8.4f} {s3:>8.4f}")
        w1, w2, w3 = weighted_vote_probs[idx]
        print(f"{'1:1:2 weighted soft voting':>26} :  {w1:>8.4f} {w2:>8.4f} {w3:>8.4f}")
        print(f"{'result':>26} :  {classification_results[final_preds[idx]]}")
        print()
        print("#####################################")

    # 5. 保存结果到文件
    save_path = "predict_result.txt"
    with open(save_path, 'w', encoding='utf-8') as f:
        for idx in range(len(features)):
            f.write(f"@task {idx+1}\n")
            f.write("Free Energy Result: \n")
            e1, e2, e3, e4, e5 = all_energies[:, idx]
            f.write(f"model 1-5 :     {e1:.4f} {e2:.4f} {e3:.4f} {e4:.4f} {e5:.4f}\n")
            f.write(f"average   :     {avg_energies[idx]:.4f}\n\n")
            f.write("Dissolution Classification Result:\n")
            for fold_idx in range(5):
                p1, p2, p3 = all_probs[fold_idx, idx]
                f.write(f"model {fold_idx+1}: {p1:.4f} {p2:.4f} {p3:.4f}\n")
            f.write("\n")
            s1, s2, s3 = soft_vote_probs[idx]
            f.write(f"               soft voting :  {s1:.4f} {s2:.4f} {s3:.4f}\n")
            w1, w2, w3 = weighted_vote_probs[idx]
            f.write(f"1:1:2 weighted soft voting :  {w1:.4f} {w2:.4f} {w3:.4f}\n")
            f.write(f"Classification Result      :  {classification_results[final_preds[idx]]}\n")
            f.write("#####################################\n")
    print(f"\nThe results have been saved to ：{save_path}")

if __name__ == "__main__":
    main()
    
