# Phân tích Dữ liệu và Dự đoán

Dự án này tập trung vào việc phân tích dữ liệu, xử lý các giá trị bị thiếu, lựa chọn đặc trưng, và xây dựng các mô hình dự đoán.

## Cấu trúc Dự án

Dự án bao gồm các thành phần chính sau:

-   `analyze_missing.py`: Tập lệnh này dùng để phân tích các thuộc tính của dữ liệu, kiểm tra xem có dữ liệu bị thiếu hay không và tính toán tỷ lệ dữ liệu bị thiếu.
-   `visualize_and_impute.py`: Tập lệnh này thực hiện trực quan hóa dữ liệu và xử lý các giá trị bị thiếu (imputation).
    -   Kết quả dữ liệu đã xử lý được lưu vào thư mục `data/`.
    -   Các biểu đồ trực quan được lưu vào thư mục `visualizations/`.
-   `chi_square_feature_selection.py`: Tập lệnh này thực hiện kiểm định Chi-square trên các đặc trưng để đánh giá mức độ quan trọng của chúng.
-   `inferential_analysis.ipynb`: Notebook Jupyter này chứa nhiều mô hình dự đoán khác nhau và thực hiện quá trình dự đoán.
    -   Kết quả dự đoán được lưu vào thư mục `output/`.

## Cài đặt

Để chạy dự án này, bạn cần cài đặt các thư viện Python được liệt kê trong `requirements.txt`.

1.  **Tạo môi trường ảo (khuyến nghị):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

## Hướng dẫn Chạy

Thực hiện các bước sau theo thứ tự để chạy toàn bộ quá trình phân tích và dự đoán của dự án:

1.  **Phân tích dữ liệu thiếu:**
    ```bash
    python analyze_missing.py
    ```

2.  **Trực quan hóa và điền dữ liệu thiếu:**
    ```bash
    python visualize_and_impute.py
    ```

3.  **Lựa chọn đặc trưng bằng Chi-square:**
    ```bash
    python chi_square_feature_selection.py
    ```

4.  **Thực hiện phân tích suy luận và dự đoán:**
    Mở notebook Jupyter và chạy tất cả các ô:
    ```bash
    jupyter notebook inferential_analysis.ipynb
    ```

Sau khi hoàn thành các bước trên, bạn có thể kiểm tra các tệp dữ liệu đã xử lý trong `data/`, các biểu đồ trong `visualizations/` và kết quả mô hình trong `output/`.
