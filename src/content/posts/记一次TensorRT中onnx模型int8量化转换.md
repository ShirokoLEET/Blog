---
title: 记一次TensorRT中onnx模型int8量化转换
published: 2026-06-15
description: '记录自己踩到的坑。'
image: ''
tags: [TensorRT,ONNXRuntime,ImageDetection]
category: 'YOLO'
draft: false 
lang: 'zh_CN'
---
受朋友邀请开发一个游戏的`YOLO`图像识别，我直接用了现成的`TensorRT`推理和其他人做的现成的`ONNX`模型。但是由于要实现良好的效果，需要实现每秒识别多帧(60FPS+)，虽然TensorRT已经帮我们优化了
很多，但是实际用下来还是会发现性能占用略高于预期，这个时候怎么优化模型就是我们要考虑的问题。在目前，我想到的优化方式只有两种:

- 1.降低模型精度
- 2.优化模型算子

由于笔者在大模型方面还是属于初学者，这边暂时先使用第一种方式进行优化。

目前手头的模型是：

**INPUTS**

| Key | Name | Tensor |
|---|---|---|
| images | images | float32[1, 3, 640, 640] |

**OUTPUTS**

| Key | Name | Tensor |
|---|---|---|
| output0 | output0 | float32[1, 9, 8400] |

第一步想到的方法自然是FP32转FP16，精度损失很小而且不需要其他操作，直接调用onnxruntime库的

```python title="onnxruntime"
float16.convert_float_to_float16(model,keep_io_types=keep_io_types)
```

经测试后，发现性能并未达到预期。检索一番后，发现TensorRT模型支持以下几种ONNX模型的精度：

| 精度 | 普通 FP32 ONNX 能否直接用 | 说明 |
|---|---|---|
| FP32 | 可以 | 默认精度。实际在 Ampere 以后可能默认走 TF32 计算。 |
| TF32 | 可以 | 不是 ONNX 权重格式，而是 FP32 计算的加速模式；TensorRT 默认启用 TF32。 |
| FP16 | 可以 | 最常用，通常是 YOLO / CNN 模型首选优化精度。 |
| BF16 | 取决于硬件 | 动态范围比 FP16 大，但尾数精度更低。 |
| INT8 | 需要量化 / 校准 | 普通 FP32 ONNX 不能无损直接变 INT8，需要校准或 Q/DQ 量化模型。 |
| FP8 | 通常需要 Q/DQ | 不是简单把 FP32 ONNX 加个参数就稳定可用，通常要显式量化。 |
| INT4 | 主要是权重量化 | TensorRT 对 INT4 是有限支持，偏向 weight-only quantization。 |
| FP4 / NVFP4 | 需要 Q/DQ，硬件限制更强 | 新版本 TensorRT 支持 FP4 类型，但实际依赖硬件和量化模型。 |

看下来，如果想要占用比FP16低的话，只有FP4和INT8。FP4目前只在Blackwell架构上支持，这里我决定将模型转换为INT8精度。

转换为INT8精度需要量化校准，而我们没有模型的数据集，只有ONNX文件，怎么办？

这里我的思路是直接将推理出来的数据重新转化为数据集，这样我们就有了一份符合这个模型的数据集。我选择的是投喂几个视频，同时采集正样本负样本。

脚本的配置区如下，核心目标是指定 `ONNX` 模型、输入视频、输出目录、置信度阈值、`NMS` 阈值、采样帧率以及每个视频保留的负样本数量：

```python title="videoDetect.py: CONFIG"
MODEL_PATH = "./model/dear.onnx"    # ONNX 模型路径

# 支持单个路径字符串，或多个路径的列表
VIDEO_PATH = [
    "./video/video1.mp4",
    "./video/video2.mp4",
    "./video/video3.mp4",
    "./video/video4.mp4",
    # 继续添加更多视频...
]

OUTPUT_DIR = "../dataset"  # 输出文件夹（自动创建）

CONF_THRESH  = 0.7    # 置信度阈值（越高越严格，建议 0.2~0.5）
IOU_THRESH   = 0.5    # NMS IoU 阈值（越低去重越激进，建议 0.4~0.6）
SAVE_CONF    = False   # True：在 txt 末尾额外保存置信度分数
IMG_QUALITY  = 100    # 保存帧的 JPEG 质量（1~100）
DETECT_FPS   = 30     # 检测帧率（None = 检测所有帧，否则按此帧率采样）

# 每个视频最多保留多少张「无目标帧」作为负样本
# 0 = 不保留，None = 保留所有无目标帧
NEG_SAMPLES_PER_VIDEO = 50
```

依赖只有 `onnxruntime`、`opencv-python` 和 `numpy`。这里使用 `onnxruntime` 读取模型做推理，再用 `OpenCV` 逐帧读取视频和保存图片：

```python title="videoDetect.py: imports"
import ast
import sys
import time
import random
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("[错误] 请先安装: pip install onnxruntime")

try:
    import cv2
except ImportError:
    sys.exit("[错误] 请先安装: pip install opencv-python")
```

模型输入前需要做 `letterbox`，保持原图比例缩放到 `640x640`，不足部分使用灰色填充。这里返回了 `scale` 和 `pad_w/pad_h`，后面把检测框坐标还原到原图时会用到：

```python title="videoDetect.py: preprocess"
def letterbox(img: np.ndarray, target_size: int):
    """等比例缩放 + 灰色填充到 target_size×target_size。"""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) / 2
    pad_h = (target_size - new_h) / 2
    top   = int(round(pad_h - 0.1))
    left  = int(round(pad_w - 0.1))

    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[top:top + new_h, left:left + new_w] = resized

    return padded, scale, (pad_w, pad_h)


def preprocess(frame_bgr: np.ndarray, input_size: int):
    """BGR 帧 → 模型输入张量。"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = frame_rgb.shape[:2]
    padded, scale, (pad_w, pad_h) = letterbox(frame_rgb, input_size)
    tensor = padded.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    return tensor, scale, (pad_w, pad_h), (orig_w, orig_h)
```

后处理部分先把模型输出中的 `xywh` 转为 `xyxy`，再按类别执行 `NMS`。脚本兼容 `[1, 4+C, N]` 和 `[1, N, 4+C]` 两种输出排列，最终返回归一化后的 `YOLO` 标注格式：

```python title="videoDetect.py: postprocess"
def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def postprocess(output, scale, pad_w, pad_h, orig_w, orig_h, conf_thresh, iou_thresh):
    """
    解析模型输出，返回检测列表: [(class_id, cx, cy, w, h, conf), ...]
    坐标归一化到原图 [0, 1]。支持 [1, 4+C, N] 和 [1, N, 4+C] 两种格式。
    """
    pred = output[0]
    if pred.shape[0] < pred.shape[1]:   # [4+C, N] → [N, 4+C]
        pred = pred.T

    boxes_raw    = pred[:, :4].copy()
    obj_conf     = pred[:, 4]
    class_scores = pred[:, 5:]

    class_ids   = np.argmax(class_scores, axis=1)
    cls_conf    = class_scores[np.arange(len(pred)), class_ids]
    confidences = obj_conf * cls_conf

    mask        = confidences >= conf_thresh
    boxes_raw   = boxes_raw[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    if len(boxes_raw) == 0:
        return []

    # letterbox 坐标 → 原图像素坐标
    boxes_raw[:, 0] = (boxes_raw[:, 0] - pad_w) / scale
    boxes_raw[:, 1] = (boxes_raw[:, 1] - pad_h) / scale
    boxes_raw[:, 2] /= scale
    boxes_raw[:, 3] /= scale

    boxes_xyxy = xywh2xyxy(boxes_raw)
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

    results = []
    for cls_id in np.unique(class_ids):
        idx  = np.where(class_ids == cls_id)[0]
        keep = nms(boxes_xyxy[idx], confidences[idx], iou_thresh)
        for k in keep:
            i = idx[k]
            x1, y1, x2, y2 = boxes_xyxy[i]
            cx = ((x1 + x2) / 2) / orig_w
            cy = ((y1 + y2) / 2) / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            results.append((int(cls_id), float(cx), float(cy), float(bw), float(bh), float(confidences[i])))
    return results
```

逐帧处理时，脚本按 `DETECT_FPS` 控制采样频率。检测到目标时保存图片和同名 `txt`，没有目标时进入负样本蓄水池，用于限制每个视频最多保留的负样本数量：

```python title="videoDetect.py: process_video 核心逻辑"
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if step > 1.0 and round(frame_idx % step) != 0:
        frame_idx += 1
        continue

    t0 = time.perf_counter()
    tensor, scale, (pad_w, pad_h), (orig_w, orig_h) = preprocess(frame, input_size)
    output = sess.run(None, {input_name: tensor})[0]
    dets   = postprocess(output, scale, pad_w, pad_h, orig_w, orig_h, conf, iou)
    elapsed     = time.perf_counter() - t0
    total_time += elapsed

    stem = f"{video_stem}_{str(frame_idx).zfill(digits)}"

    if dets:
        # ── 正样本 ────────────────────────────────────────────────────────
        img_path = output_dir / f"{stem}.jpg"
        txt_path = output_dir / f"{stem}.txt"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, img_quality])
        with open(txt_path, "w") as f:
            for cls_id, cx, cy, bw, bh, conf_score in dets:
                if save_conf:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf_score:.4f}\n")
                else:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        pos_saved   += 1
        total_boxes += len(dets)
        print(f"  {str(frame_idx).zfill(digits):>{digits}}  {len(dets):>6}  {elapsed*1000:>6.1f} ms  {stem}.jpg")

    else:
        # ── 负样本：蓄水池抽样 ────────────────────────────────────────────
        if neg_samples_per_video != 0:
            neg_seen += 1
            if save_all_neg:
                neg_reservoir.append((frame_idx, frame.copy()))
            elif len(neg_reservoir) < neg_samples_per_video:
                neg_reservoir.append((frame_idx, frame.copy()))
            else:
                # 以 n/neg_seen 的概率替换蓄水池中的随机位置
                j = random.randint(0, neg_seen - 1)
                if j < neg_samples_per_video:
                    neg_reservoir[j] = (frame_idx, frame.copy())

    frame_idx += 1
```

负样本最终也会写出图片和空的同名 `txt`。这样生成的数据目录里，正样本帧有 `YOLO` 标注，负样本帧为空标注文件：

```python title="videoDetect.py: 写出负样本"
neg_saved = 0
for fidx, fimg in neg_reservoir:
    neg_stem = f"{video_stem}_{str(fidx).zfill(digits)}"
    cv2.imwrite(str(output_dir / f"{neg_stem}.jpg"), fimg,
                [cv2.IMWRITE_JPEG_QUALITY, img_quality])
    open(output_dir / f"{neg_stem}.txt", "w").close()   # 空标注文件
    neg_saved += 1
```

主流程只加载一次模型，然后复用同一个 `InferenceSession` 处理多个视频。输入尺寸直接从模型输入元信息读取，类别名尝试从 `custom_metadata_map` 的 `names` 字段读取：

```python title="videoDetect.py: run"
def run(model_path, video_paths, output_dir, conf, iou, save_conf,
        img_quality, detect_fps, neg_samples_per_video):

    # 统一转为列表
    if isinstance(video_paths, (str, Path)):
        video_paths = [video_paths]
    video_paths = [Path(p) for p in video_paths]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载模型（只加载一次） ────────────────────────────────────────────────
    print(f"[INFO] 加载模型: {model_path}")
    providers  = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess       = ort.InferenceSession(model_path, providers=providers)
    inp_meta   = sess.get_inputs()[0]
    out_meta   = sess.get_outputs()[0]
    input_name = inp_meta.name
    _, _, h, w = inp_meta.shape
    input_size = h
    print(f"[INFO] 推理后端: {sess.get_providers()[0]}")

    try:
        raw_meta    = sess.get_modelmeta().custom_metadata_map
        class_names = ast.literal_eval(raw_meta.get("names", "{}"))
    except Exception:
        class_names = {}

    num_cls = (out_meta.shape[1] - 4) if out_meta.shape[1] > out_meta.shape[2] else (out_meta.shape[2] - 4)
    print(f"[INFO] 输入尺寸: {input_size}×{input_size}  |  类别数: {len(class_names) or num_cls}")
    if class_names:
        print(f"[INFO] 类别: {class_names}")
    neg_desc = "全部" if neg_samples_per_video is None else (
               "不保留" if neg_samples_per_video == 0 else f"每视频最多 {neg_samples_per_video} 张")
    print(f"[INFO] 负样本策略: {neg_desc}")
    print(f"[INFO] 输出目录: {output_dir}\n")
    print(f"[INFO] 共 {len(video_paths)} 个视频待处理\n")
```

最后保留一个最简单的入口，直接运行脚本即可根据配置区生成数据集：

```python title="videoDetect.py: main"
if __name__ == "__main__":
    run(MODEL_PATH, VIDEO_PATH, OUTPUT_DIR, CONF_THRESH, IOU_THRESH,
        SAVE_CONF, IMG_QUALITY, DETECT_FPS, NEG_SAMPLES_PER_VIDEO)
```

这个脚本生成的数据集结构非常直接：

```text title="../dataset"
video1_000001.jpg
video1_000001.txt
video1_000030.jpg
video1_000030.txt
...
```

其中 `.jpg` 是从视频中抽取的帧，`.txt` 是对应的 `YOLO` 格式标注。正样本的 `txt` 内容类似：

```text title="正样本 txt"
0 0.521337 0.487612 0.124830 0.226401
```

负样本的 `txt` 文件为空。这样做的好处是后续无论是用于继续训练，还是用于 `INT8` 校准集，都可以直接复用这一批和实际使用场景更接近的图像。

利用这个脚本，我们可以快速的生成出数据集。
之后，我们就能利用数据集开始int8量化校准。

## 使用 ONNX Runtime 做静态 INT8 量化

这次量化使用的是 `onnxruntime.quantization.quantize_static`，模型采用 `QDQ` 格式。原始模型的精确信息如下：

```text title="dear.onnx"
input:  images  tensor(float)  [1, 3, 640, 640]
output: output0 tensor(float)  [1, 9, 8400]
classes: {0: 'body', 1: 'head', 2: 'teammate', 3: 'breakable', 4: 'dodge'}
has_objectness: False
```

数据集最终统计如下：

```text title="../dataset"
jpg: 1708
txt: 1708
nonempty_txt: 1258
empty_txt: 450
```

### 基础量化脚本

量化脚本需要实现一个 `CalibrationDataReader`，按照前面推理脚本完全相同的 `letterbox + RGB + NCHW + /255.0` 预处理方式输出模型输入：

```python title="CalibrationDataReader"
class ImageReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str, input_size: int, reported_len: int | None = None):
        self.image_paths = image_paths
        self.input_name = input_name
        self.input_size = input_size
        self.reported_len = reported_len if reported_len is not None else len(image_paths)
        self.start_index = 0
        self.end_index = len(image_paths)
        self.index = self.start_index

    def get_next(self):
        if self.index >= self.end_index:
            return None
        path = self.image_paths[self.index]
        self.index += 1
        tensor, *_ = preprocess_image(path, self.input_size)
        return {self.input_name: tensor}

    def __len__(self):
        return self.reported_len

    def set_range(self, start_index: int, end_index: int):
        if start_index < 0 or start_index > len(self.image_paths) or start_index > end_index:
            raise ValueError(f"invalid range: start_index={start_index}, end_index={end_index}")
        self.start_index = start_index
        self.end_index = min(end_index, len(self.image_paths))
        self.index = self.start_index
```

实际调用 `quantize_static` 的核心参数如下：

```python title="quantize_static"
quantize_static(
    model_input=tmp_pre,
    model_output=output,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    op_types_to_quantize=["Conv", "MatMul"],
    per_channel=True,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_exclude=None,
    calibrate_method=CalibrationMethod.Percentile,
    calibration_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "CalibTensorRangeSymmetric": True,
        "MinimumRealRange": 0.0000001,
        "CalibPercentile": 99.999,
        "CalibStridedMinMax": 16,
    },
)
```

这里采用：

- `QDQ`
- `Conv` 和 `MatMul`
- `per_channel=True`
- activation 使用 `QInt8`
- weight 使用 `QInt8`
- calibration 使用 `Percentile`
- percentile 取 `99.999`

## Percentile 校准为什么会吃内存

一开始直接使用 512 张图做 `Percentile` 校准时，进程出现了接近 120GB 的常驻内存，占满物理内存后进入 swap，系统表现为长时间卡住。

原因在 `onnxruntime.quantization.calibrate.py` 里。`Percentile` 使用的是 `HistogramCalibrater`，它会把需要校准的中间张量追加为模型输出。每跑一张图，`infer_session.run(None, inputs)` 返回的中间张量都会进入 `self.intermediate_outputs`。这些输出是 CPU 侧的 `numpy.ndarray`，之后再交给 `HistogramCollector` 做直方图和阈值统计。

也就是说，哪怕推理用了 `CUDAExecutionProvider`，`Percentile` 的统计阶段仍然会把大量中间结果拉回 CPU 内存。显存不高不代表任务没在跑；这里的瓶颈在 CPU 内存和 NumPy 直方图统计。

ONNX Runtime 的 `quantize_static` 有一个参数名叫 `CalibStridedMinMax`，虽然名字里有 `MinMax`，但在当前安装版本中，它会让 `quantize_static` 分段调用 `calibrator.collect_data()`：

```python title="onnxruntime.quantization.quantize.py"
stride = extra_options.get("CalibStridedMinMax", None)
if stride:
    total_data_size = len(calibration_data_reader)
    if total_data_size % stride != 0:
        raise ValueError(f"Total data size ({total_data_size}) is not divisible by stride size ({stride}).")

    for start in range(0, total_data_size, stride):
        end_index = start + stride
        calibration_data_reader.set_range(start_index=start, end_index=end_index)
        calibrator.collect_data(calibration_data_reader)
else:
    calibrator.collect_data(calibration_data_reader)
```

所以脚本里实现了 `__len__()` 和 `set_range()`，再传入：

```python
"CalibStridedMinMax": 16
```

数据集一共有 1708 张图，不能被 16 整除。为保持 16 张一块，同时不丢最后 12 张图，脚本让 `__len__()` 返回 1712，最后一个分块在 `set_range()` 中截断到真实图片数量：

```python
reported_len = int(math.ceil(len(images) / calib_stride) * calib_stride)
reader = ImageReader(images, layout.input_name, layout.input_size, reported_len=reported_len)
```

这样 ONNX Runtime 会按 1712 张计算分块数量，实际读取到第 1708 张结束。

全量分块校准结果：

```text title="quant_percentile1708_stride16.log"
calibration_images: 1708
calib_stride: 16
reader_reported_len: 1712
elapsed: 1:04:47
Maximum resident set size: 7587072 KB
Swaps: 0
Exit status: 0
```

这说明分块校准后，内存峰值约 7.59GB，没有再出现 120GB 内存占用。

## 这些模型分别代表什么

为了避免后面对比表看起来像一堆随机文件名，这里先把模型命名和来源讲清楚，但不把每个模型都贴一遍命令。

通用量化命令如下，后面几个模型只是替换 `--method`、`--calib-limit`、`--calib-stride` 和 `--output`：

```bash
python int8_quant_eval.py quantize \
  --source ./model/dear.onnx \
  --dataset ./dataset \
  --calib-limit 64 \
  --calib-stride 16 \
  --output ./model/dear_int8_s8s8_pc_percentile99999_64_stride16.onnx \
  --method percentile \
  --activation-type qint8 \
  --percentile 99.999
```

模型命名规则大致如下：

```text
dear_int8_s8s8_pc_percentile99999_64_stride16.onnx
     │    │    │       │              │    │
     │    │    │       │              │    └─ 每 16 张图做一次分块 collect_data
     │    │    │       │              └──── 校准图片数量，这里是 64 张
     │    │    │       └─────────────────── Percentile 校准，百分位 99.999
     │    │    └─────────────────────────── per_channel=True
     │    └──────────────────────────────── activation=QInt8，weight=QInt8
     └───────────────────────────────────── 静态 INT8 QDQ 量化模型
```

各字段含义如下：

- `int8`：使用 ONNX Runtime 静态量化生成的 INT8 QDQ 模型。
- `s8s8`：activation 使用 `QInt8`，weight 使用 `QInt8`。
- `pc`：`per_channel=True`，权重量化按通道做 scale。
- `minmax`：使用 `CalibrationMethod.MinMax`。
- `percentile99999`：使用 `CalibrationMethod.Percentile`，并设置 `"CalibPercentile": 99.999`。
- `64` / `512` / `1708`：校准时实际使用的图片数量。
- `stride16`：通过 `"CalibStridedMinMax": 16` 每 16 张图分块收集中间张量，控制内存峰值。
- `_trt`：额外折叠了 `INT32` bias 的 `DequantizeLinear`，用于绕过 TensorRT 对 int32 bias DQ 的解析问题。

几个模型的区别如下：

| 模型 | 怎么来的 | 代表什么 |
|---|---|---|
| `dear_int8_s8s8_pc_minmax_all.onnx` | 使用全部 1708 张图做 `MinMax` 校准，其他量化设置和 Percentile 模型一致 | MinMax 路线的对照组，用来验证 MinMax 是否适合这个 YOLO 模型 |
| `dear_sint8_zp0.onnx` | 目录里原本已有的旧 INT8 QDQ 模型；从文件结构能确认它是 ONNX Runtime quant 产物，包含 349 个 `DequantizeLinear`、174 个 `QuantizeLinear`、87 个 int32 bias DQ | 旧 INT8 基线模型，用来判断新方案是否真的变好 |
| `dear_int8_s8s8_pc_percentile99999_64_stride16.onnx` | 使用 64 张图做 `Percentile 99.999` 校准，每 16 张分块 | 本次检测指标最好的普通 QDQ INT8 模型 |
| `dear_int8_s8s8_pc_percentile99999_512_stride16.onnx` | 使用 512 张图做 `Percentile 99.999` 校准，每 16 张分块 | 用来验证更多校准图片是否带来更好检测指标 |
| `dear_int8_s8s8_pc_percentile99999_1708_stride16.onnx` | 使用全部 1708 张图做 `Percentile 99.999` 校准，每 16 张分块；最后一个分块实际只读取剩余 12 张图 | 用来验证全量分块 Percentile 校准是否可行，以及全量分布下的精度表现 |
| `dear_int8_s8s8_pc_percentile99999_64_stride16_trt.onnx` | 在 64 张 Percentile 模型基础上折叠 87 个 int32 bias DQ，将 bias 还原为 FP32 initializer | TensorRT 使用的版本；ONNX Runtime 检测结果和 64 张普通 QDQ 模型一致 |

全量 1708 张分块模型的校准日志如下：

```text
calibration_images: 1708
calib_stride: 16
reader_reported_len: 1712
elapsed: 1:04:47
Maximum resident set size: 7587072 KB
Swaps: 0
Exit status: 0
```

其中 `reader_reported_len=1712` 只是为了让 ONNX Runtime 能按 16 张分块迭代；真实图片数量仍然是 1708，最后一个分块会截断到真实图片末尾。

## 量化结果对比

这里用 FP32 `dear.onnx` 作为参考，评估各个 INT8 模型与 FP32 检测结果的一致性。评估指标使用：

- `precision_to_fp32`
- `recall_to_fp32`
- `f1_to_fp32`
- `tp`
- `fp`
- `fn`
- `mean_iou_on_matches`
- `mean_score_abs_error_on_matches`

### conf=0.5

| 模型 | precision | recall | F1 | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| dear_int8_s8s8_pc_minmax_all.onnx | 0.9984 | 0.3633 | 0.5328 | 630 | 1 | 1104 |
| dear_sint8_zp0.onnx | 0.9982 | 0.9544 | 0.9758 | 1655 | 3 | 79 |
| dear_int8_s8s8_pc_percentile99999_64_stride16.onnx | 0.9735 | 0.9942 | 0.9837 | 1724 | 47 | 10 |
| dear_int8_s8s8_pc_percentile99999_512_stride16.onnx | 0.9664 | 0.9960 | 0.9810 | 1727 | 60 | 7 |
| dear_int8_s8s8_pc_percentile99999_1708_stride16.onnx | 0.9659 | 0.9960 | 0.9807 | 1727 | 61 | 7 |

### conf=0.7

| 模型 | precision | recall | F1 | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| dear_int8_s8s8_pc_minmax_all.onnx | 1.0000 | 0.0069 | 0.0138 | 11 | 0 | 1574 |
| dear_sint8_zp0.onnx | 0.9992 | 0.7754 | 0.8732 | 1229 | 1 | 356 |
| dear_int8_s8s8_pc_percentile99999_64_stride16.onnx | 0.9973 | 0.9218 | 0.9580 | 1461 | 4 | 124 |
| dear_int8_s8s8_pc_percentile99999_512_stride16.onnx | 0.9986 | 0.8789 | 0.9349 | 1393 | 2 | 192 |
| dear_int8_s8s8_pc_percentile99999_1708_stride16.onnx | 0.9986 | 0.8726 | 0.9313 | 1383 | 2 | 202 |

从检测结果看，`dear_int8_s8s8_pc_percentile99999_64_stride16.onnx` 在 `conf=0.5` 和 `conf=0.7` 下都是 F1 最高的模型。全量 1708 张分块模型的输出张量误差更低，但检测指标略低于 64 张分块模型。

所以当前推荐使用：

```text
dear_int8_s8s8_pc_percentile99999_64_stride16.onnx
```

### 为什么 MinMax 校准效果差

MinMax 和 Percentile 的量化模型使用的是同一批图片、同一套预处理、同样的 `QDQ / S8S8 / per_channel / Conv+MatMul` 量化设置。差异只在 calibration method。

实际对比结果很直观：MinMax 不是误检多，而是漏检非常多。

```text title="conf=0.5"
FP32 reference boxes: 1734
MinMax pred_boxes: 631
MinMax tp=630 fp=1 fn=1104
MinMax recall=0.3633
MinMax F1=0.5328

Percentile 64 pred_boxes: 1771
Percentile 64 tp=1724 fp=47 fn=10
Percentile 64 recall=0.9942
Percentile 64 F1=0.9837
```

```text title="conf=0.7"
FP32 reference boxes: 1585
MinMax pred_boxes: 11
MinMax tp=11 fp=0 fn=1574
MinMax recall=0.0069
MinMax F1=0.0138

Percentile 64 pred_boxes: 1465
Percentile 64 tp=1461 fp=4 fn=124
Percentile 64 recall=0.9218
Percentile 64 F1=0.9580
```

从误差指标看，MinMax 的输出张量整体偏差也明显更大：

| 模型 | mean_output_mae | mean_output_rmse | mean_box_mae | mean_score_abs_error_on_matches |
|---|---:|---:|---:|---:|
| MinMax all | 4.5170 | 12.3715 | 10.1633 | 0.3738 |
| Percentile 64 | 1.9250 | 5.5031 | 4.3314 | 0.0313 |

ONNX Runtime 的 MinMax 校准逻辑会给每个待校准张量收集 `ReduceMin` / `ReduceMax`。在开启对称范围时，最终量化范围会受最大绝对值控制：

```python title="MinMax 对称范围"
max_absolute_value = np.nanmax([np.abs(min_value_array), np.abs(max_value_array)], axis=0)
range = (-max_absolute_value, max_absolute_value)
```

这个策略的问题是：只要某个中间张量里出现极少量幅度很大的值，整个张量的量化范围就会被这些极值拉大。INT8 的可表示档位固定，范围被拉大以后，主要分布区域的量化间隔也会变大，常见特征值、检测框坐标和分类分数都会变得更粗。

这次模型的 MinMax 结果正好体现了这个问题：

- `pred_boxes` 大幅减少，说明大量 FP32 中能过阈值的框在 MinMax INT8 后没有过阈值。
- `precision` 仍然接近 1，说明剩下来的框多数还是对的。
- `recall` 很低，说明 MinMax 的主要问题是把大量有效框压没了。
- `mean_score_abs_error_on_matches` 从 Percentile 的 `0.0313` 变成 MinMax 的 `0.3738`，说明匹配上的框也出现了明显的分数偏移。
- `mean_box_mae` 从 Percentile 的 `4.3314` 变成 MinMax 的 `10.1633`，说明坐标输出也比 Percentile 偏得更多。

Percentile 的做法是对张量分布做直方图统计，然后按百分位裁掉极少数极值。这里使用的是：

```python
"CalibPercentile": 99.999
```

也就是保留绝大多数数值分布，把极少量尾部极值排除在量化范围之外。对于这类 YOLO 检测模型，检测结果更依赖主体分布里的特征、坐标和分类分数是否稳定，而不是保留少量极端激活值。因此 Percentile 的检测结果比 MinMax 更接近 FP32。

结论是：这个模型不适合用全局 MinMax 做 INT8 calibration。MinMax 的高 precision 是因为它只留下了少量高置信框，不代表整体精度更好；从检测任务看，MinMax 的 recall 和 F1 都明显不可用。Percentile 64 分块模型是当前指标最好的方案。

## TensorRT 解析 int32 bias DequantizeLinear 报错

把上面的 QDQ 模型交给 TensorRT 11.0.0 解析时，遇到下面这个报错：

```text title="trtexec"
[E] Error[4]: ITensor::getDimensions: Error Code 4: API Usage Error
(model.0.conv.bias_DequantizeLinear: input has type Int32 but must have type FP8, FP4, Int4, or Int8.)

While parsing node number 1 [DequantizeLinear -> "model.0.conv.bias"]:
Invalid Node - model.0.conv.bias_DequantizeLinear
```

检查 ONNX 图后可以看到，ONNX Runtime 静态量化会把 bias 量化成 `INT32`，再通过 `DequantizeLinear` 还原成 FP32 bias：

```text
DequantizeLinear model.0.conv.bias_DequantizeLinear
inputs:
  model.0.conv.bias_quantized
  model.0.conv.bias_quantized_scale
  model.0.conv.bias_quantized_zero_point
output:
  model.0.conv.bias

model.0.conv.bias_quantized: INT32 [32]
model.0.conv.bias_quantized_scale: FLOAT [32]
model.0.conv.bias_quantized_zero_point: INT32 [32]
```

新生成的 64 张分块模型中，这类 `INT32` bias `DequantizeLinear` 一共有 87 个：

```text
DequantizeLinear: 349
QuantizeLinear: 174
Conv: 88
int32 bias DequantizeLinear: 87
```

而之前能给 TensorRT 使用的 `dear_sint8_zp0_trt.onnx` 里已经没有这些 int32 bias DQ：

```text
DequantizeLinear: 262
QuantizeLinear: 174
Conv: 88
int32 bias DequantizeLinear: 0
```

对比后可以确定：TensorRT 版本需要把 int32 bias DQ 折叠回普通 FP32 bias initializer。

折叠公式就是标准的 `DequantizeLinear`：

```python
bias = (bias_quantized.astype(np.float32) - zero_point.astype(np.float32)) * scale.astype(np.float32)
```

处理脚本如下：

```python title="make_trt_compatible_qdq.py"
def fold_int32_bias_dq(input_path: Path, output_path: Path) -> int:
    model = onnx.load(str(input_path), load_external_data=False)
    initializers = {item.name: item for item in model.graph.initializer}

    folded = 0
    remove_node_names: set[str] = set()
    remove_initializer_names: set[str] = set()
    new_initializers = []

    for node in model.graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) < 2 or len(node.output) != 1:
            continue
        quantized_name = node.input[0]
        scale_name = node.input[1]
        zero_point_name = node.input[2] if len(node.input) >= 3 else None
        quantized = initializers.get(quantized_name)
        scale = initializers.get(scale_name)
        zero_point = initializers.get(zero_point_name) if zero_point_name else None
        if quantized is None or scale is None:
            continue
        if quantized.data_type != onnx.TensorProto.INT32:
            continue
        if not quantized_name.endswith(".bias_quantized"):
            continue

        quantized_arr = numpy_helper.to_array(quantized).astype(np.float32)
        scale_arr = numpy_helper.to_array(scale).astype(np.float32)
        if zero_point is not None:
            zero_point_arr = numpy_helper.to_array(zero_point).astype(np.float32)
        else:
            zero_point_arr = np.array(0, dtype=np.float32)

        bias_arr = (quantized_arr - zero_point_arr) * scale_arr
        bias_arr = bias_arr.astype(np.float32)
        new_initializers.append(numpy_helper.from_array(bias_arr, node.output[0]))

        remove_node_names.add(node.name)
        remove_initializer_names.add(quantized_name)
        remove_initializer_names.add(scale_name)
        if zero_point_name:
            remove_initializer_names.add(zero_point_name)
        folded += 1

    if folded == 0:
        raise RuntimeError(f"no int32 bias DequantizeLinear nodes folded in {input_path}")

    kept_nodes = [node for node in model.graph.node if node.name not in remove_node_names]
    kept_initializers = [
        item for item in model.graph.initializer if item.name not in remove_initializer_names
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    model.graph.initializer.extend(new_initializers)

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return folded
```

实际处理时只需要对 64 张 Percentile 模型运行这个折叠脚本，输入普通 QDQ 模型，输出 `_trt.onnx`。这里不再重复贴生成命令，避免把环境路径写进记录里。

处理结果：

```text
folded_int32_bias_dq=87
```

处理后的模型结构：

```text
path: model/dear_int8_s8s8_pc_percentile99999_64_stride16_trt.onnx
size: 9945209
DequantizeLinear: 262
QuantizeLinear: 174
Conv: 88
int32 DequantizeLinear: 0
model.0.conv.bias: FLOAT [32]
```

这个 `_trt.onnx` 和原始 64 张分块 QDQ 模型在 ONNX Runtime 下的检测结果完全一致：

```text title="conf=0.7"
model: dear_int8_s8s8_pc_percentile99999_64_stride16.onnx
tp=1461 fp=4 fn=124 precision=0.9973 recall=0.9218 f1=0.9580

model: dear_int8_s8s8_pc_percentile99999_64_stride16_trt.onnx
tp=1461 fp=4 fn=124 precision=0.9973 recall=0.9218 f1=0.9580
```

因此用于 TensorRT 的最终文件应当是：

```text
dear_int8_s8s8_pc_percentile99999_64_stride16_trt.onnx
```

而不是未折叠 bias DQ 的：

```text
dear_int8_s8s8_pc_percentile99999_64_stride16.onnx
```

最后
`text
trtexec --onnx=.\model\dear_int8.onnx --saveEngine=.\model\dear_int8.engine --builderOptimizationLevel=5 --avgTiming=16 --timingCacheFile=.\model\dear_int8.timing.cache --sparsity=enable --tilingOptimizationLevel=3
`
即可获得一份int8量化成功的engine模型，GPU占用相对降低50%。
