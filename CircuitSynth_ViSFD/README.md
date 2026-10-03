# CircuitSynth ViSFD Pipeline

Repository này chứa pipeline cài đặt và thực nghiệm CircuitSynth trên dữ liệu tiếng Việt ViSFD.

## 1. Cấu trúc thư mục

```text
.
├── run.py
├── config.yaml
├── pyproject.toml
├── requirements.lock
├── src/
│   ├── data.py
│   ├── schema.py
│   ├── teacher.py
│   ├── verifier.py
│   ├── circuits.py
│   ├── student.py
│   ├── fsa.py
│   ├── evaluate.py
│   └── checkpoint.py
├── tests/
│   ├── test_core.py
│   └── test_pipeline.py
├── notebooks/
│   └── CircuitSynth_ViSFD.ipynb
└── artifacts/              # được tạo sau khi chạy
```

## 2. Yêu cầu môi trường

- Python 3.10 trở lên
- NVIDIA GPU có CUDA để chạy Teacher, Student và bitsandbytes
- Kết nối Internet để tải dataset và model từ Hugging Face

Pipeline hiện sử dụng:

- Dataset: `visolex/ViSFD`
- Teacher: `Qwen/Qwen2.5-1.5B-Instruct`
- Student: `Qwen/Qwen2.5-0.5B-Instruct`
- 4-bit NF4 quantization
- QLoRA cho Student

Các tham số chính được đặt trong `config.yaml`.

## 3. Cài đặt

### Windows PowerShell

Tạo môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài PyTorch có CUDA phù hợp với máy, sau đó cài các thư viện của project:

```powershell
pip install -e ".[full,test]"
```

### Linux / Google Colab / Kaggle

```bash
python -m pip install --upgrade pip
pip install -e ".[full,test]"
```

Có thể kiểm tra nhanh môi trường bằng:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 4. Dataset ViSFD

Dataset được tải từ Hugging Face:

- Dataset: `visolex/ViSFD`
- URL: `https://huggingface.co/datasets/visolex/ViSFD`
- Train: 700 mẫu
- Validation: 100 mẫu
- Test: 200 mẫu

Mỗi record được chuẩn hóa thành một semantic plan chứa các cặp aspect--sentiment từ annotation của ViSFD.

Dữ liệu được tải trực tiếp trong pipeline và không phụ thuộc vào đường dẫn cục bộ trên máy cá nhân.

## 5. Chạy pipeline

Chạy toàn bộ pipeline từ đầu:

```bash
python run.py run
```

Trước khi bắt đầu Stage 1, chương trình sẽ kiểm tra CUDA, các thư viện cần thiết, khả năng truy cập dataset/model và backend của probabilistic circuit.

Pipeline gồm 8 stage:

1. Chuẩn bị dataset ViSFD và schema.
2. Sinh silver data bằng Teacher và kiểm tra bằng verifier.
3. Fit probabilistic semantic prior từ các semantic plan hợp lệ.
4. Tối ưu phân phối theo soft constraints.
5. Sample semantic plans.
6. Fine-tune Student bằng QLoRA.
7. Sinh dữ liệu tiếng Việt với constrained decoding.
8. Đánh giá và lưu metrics.

Kết quả của từng stage được lưu trong thư mục `artifacts/`.

## 6. Tiếp tục khi chương trình bị dừng

Nếu quá trình chạy bị dừng giữa chừng, dùng:

```bash
python run.py resume
```

Pipeline sẽ kiểm tra checkpoint và tiếp tục từ stage chưa hoàn thành.

Kiểm tra trạng thái các stage:

```bash
python run.py status
```

## 7. Chạy riêng một stage

Có thể chạy riêng Stage 1 đến Stage 8:

```bash
python run.py stage 1
python run.py stage 2
python run.py stage 3
```

Ví dụ chạy riêng Stage 8:

```bash
python run.py stage 8
```

Stage được chạy riêng vẫn cần các artifact đầu vào của những stage trước.

## 8. Cấu hình ViSFD

Cấu hình mặc định nằm trong `config.yaml`.

Cấu hình hiện tại sử dụng:

- ViSFD train: 700 mẫu
- ViSFD validation: 100 mẫu
- ViSFD test: 200 mẫu
- Silver accepted: 700
- Sampled plans: 200
- Student epochs: 1
- Micro batch size: 1
- Gradient accumulation: 8
- Maximum sequence length: 512
- Learning rate: `2e-5`
- QLoRA rank: 8
- QLoRA alpha: 16
- QLoRA dropout: 0.05
- Seed: 42

Cấu hình thực tế của lần chạy được lưu tại:

```text
artifacts/resolved_config.yaml
```

## 9. Kết quả đầu ra

Sau khi chạy, các artifact chính nằm trong:

```text
artifacts/
├── data/          # dữ liệu ViSFD đã chuẩn hóa và schema
├── silver/        # silver data đã accepted/rejected
├── circuits/      # semantic prior và kết quả optimization
├── plans/         # sampled semantic plans
├── student/       # Student adapter và checkpoint
├── outputs/       # dữ liệu sinh bởi các variant
├── metrics/       # metrics và báo cáo đánh giá
└── checkpoints/   # trạng thái từng stage
```

Các file chính:

```text
artifacts/data/manifests.json
artifacts/silver/quota_report.json
artifacts/circuits/projection_report.json
artifacts/plans/sampling_report.json
artifacts/student/training_report.json
artifacts/metrics/metrics.json
```

Student adapter sau khi fine-tune được lưu trong:

```text
artifacts/student/final/
```

Nếu adapter không được đóng gói trực tiếp cùng repository, có thể cung cấp đường dẫn tải model tương ứng.

## 10. Jupyter Notebook

Notebook thực nghiệm nằm tại:

```text
notebooks/CircuitSynth_ViSFD.ipynb
```

Notebook sử dụng trực tiếp các module trong `src/` và gồm các phần:

1. Cài đặt thư viện và cấu hình môi trường.
2. Tải và đọc ViSFD.
3. Khám phá và tiền xử lý dữ liệu.
4. Cài đặt hoặc huấn luyện mô hình.
5. Đánh giá trên tập test.
6. So sánh với baseline và các ablation.
7. Phân tích lỗi.
8. Demo trên dữ liệu tiếng Việt mới.

Notebook sử dụng path tương đối và được thiết kế để chạy trên Google Colab hoặc Kaggle sau khi repository được upload hoặc giải nén vào runtime.

## 11. Chạy test

```bash
pytest
```

Hoặc:

```bash
python -m unittest discover -s tests -v
```

Các test không cần tải dataset hoặc model và không ghi đè kết quả thực nghiệm trong `artifacts/`.