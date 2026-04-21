import os
import json
import torch
import shutil
from PIL import Image, ImageEnhance
from pathlib import Path
from transformers import DetrForObjectDetection, DetrImageProcessor
from torchvision.ops import nms
from safetensors.torch import load_file, save_file

# ==========================================
# 1. 配置 
# ==========================================
soup_targets = [
    "./best_model_hw2/model.safetensors",
    "./checkpoints_hw2/epoch_60/model.safetensors",
    "./checkpoints_hw2/epoch_70/model.safetensors",
]
OUTPUT_DIR = "./souped_model_final_v2"
TEST_IMG_DIR = "./data/test"
FINAL_JSON = "pred.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 戰術參數：低門檻+NMS
PRED_THRESHOLD = 0.005
IOU_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 執行權重合併
# ==========================================
print("🚀 正在執行 70 輪權重合併...")
# 使用 dict 確保即便路徑重複，模型也只會載入一次不同的權重檔
unique_targets = list(
    set([os.path.abspath(p) for p in soup_targets if os.path.exists(p)])
)
state_dicts = [load_file(p) for p in unique_targets]

print(f"📊 實際參與合併的模型數量: {len(state_dicts)}")

soup_sd = {}
for key in state_dicts[0].keys():
    soup_sd[key] = torch.stack([sd[key] for sd in state_dicts]).mean(dim=0)

save_file(soup_sd, os.path.join(OUTPUT_DIR, "model.safetensors"))
shutil.copy("./best_model_hw2/config.json", os.path.join(OUTPUT_DIR, "config.json"))
shutil.copy(
    "./best_model_hw2/preprocessor_config.json",
    os.path.join(OUTPUT_DIR, "preprocessor_config.json"),
)
print("✅ Model Soup 完成。")

# ==========================================
# 3. TTA 最終推論 (帶強制統計)
# ==========================================
print("🔍 啟動 TTA + NMS 推論...")
processor = DetrImageProcessor.from_pretrained(OUTPUT_DIR)
model = DetrForObjectDetection.from_pretrained(OUTPUT_DIR).to(DEVICE)
model.eval()

test_images = sorted(list(Path(TEST_IMG_DIR).glob("*.png")))
all_results = []
total_imgs = len(test_images)

with torch.no_grad():
    for i, img_path in enumerate(test_images):
        orig_image = Image.open(img_path).convert("RGB")
        W, H = orig_image.size

        # 3 種視角：原圖、調亮、放大
        views = [
            (orig_image, 1.0),
            (ImageEnhance.Brightness(orig_image).enhance(1.2), 1.0),
            (orig_image.resize((int(W * 1.1), int(H * 1.1)), Image.BILINEAR), 1.1),
        ]

        c_boxes, c_scores, c_labels = [], [], []
        for view_img, scale in views:
            inputs = processor(images=view_img, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            target_sizes = torch.tensor([[view_img.height, view_img.width]]).to(DEVICE)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=PRED_THRESHOLD
            )

            for res in results:
                # 座標還原並依尺度縮放
                c_boxes.append(res["boxes"].cpu() / scale)
                c_scores.append(res["scores"].cpu())
                c_labels.append(res["labels"].cpu())

        if not c_boxes or len(torch.cat(c_boxes)) == 0:
            continue

        boxes, scores, labels = (
            torch.cat(c_boxes),
            torch.cat(c_scores),
            torch.cat(c_labels),
        )

        # 類別 NMS 處理
        final_idxs = []
        for class_id in range(10):
            idxs = (labels == class_id).nonzero().squeeze(1)
            if idxs.numel() == 0:
                continue
            keep = nms(boxes[idxs], scores[idxs], IOU_THRESHOLD)
            final_idxs.append(idxs[keep])

        if not final_idxs:
            continue
        for idx in torch.cat(final_idxs):
            x1, y1, x2, y2 = boxes[idx].tolist()
            all_results.append(
                {
                    "image_id": int(img_path.stem),
                    "bbox": [
                        round(x1, 3),
                        round(y1, 3),
                        round(max(0, x2 - x1), 3),
                        round(max(0, y2 - y1), 3),
                    ],
                    "score": round(scores[idx].item(), 6),
                    "category_id": int(labels[idx].item()) + 1,  # 1-indexed
                }
            )

        # 每 1000 張印一次進度
        if (i + 1) % 1000 == 0:
            print(f"  [Progress] {i + 1}/{total_imgs} images processed...")

# ==========================================
# 4. 強制印出統計並存檔
# ==========================================
with open(FINAL_JSON, "w", encoding="utf-8") as f:
    json.dump(all_results, f)

print("\n" + "=" * 30)
print(f"🏁 任務成功完成！")
print(f"📁 輸出檔案: {os.path.abspath(FINAL_JSON)}")
print(f"📦 總計產出預測框數量: {len(all_results)}")
print(f"⏰ 預計平均每張圖框數: {len(all_results)/total_imgs:.2f}")
print("=" * 30)
