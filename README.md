# 📊 Dự Báo GDP Thế Giới - Hệ Hỗ Trợ Quyết Định

## 📝 Giới thiệu

Đây là dự án dự báo GDP (Tổng sản phẩm quốc nội) sử dụng nhiều mô hình Machine Learning và Deep Learning khác nhau. Dự án được xây dựng nhằm so sánh hiệu suất của các thuật toán dự báo và hỗ trợ ra quyết định dựa trên dữ liệu GDP thế giới.

## 📁 Cấu trúc dự án

```
├── app_tkinter.py                    # Ứng dụng GUI với Tkinter
├── World GDP Dataset.csv             # Bộ dữ liệu GDP thế giới
│
├── 📈 Mô hình hồi quy tuyến tính
│   ├── linear_agression_1.ipynb
│   ├── linear_agression_2.ipynb
│   ├── linear_agression_3 copy 2.ipynb
│   └── linear_agression_4.ipynb
│
├── 📉 Mô hình chuỗi thời gian
│   ├── arima.ipynb
│   ├── ARIMA1.ipynb
│   ├── ARIMA2.ipynb
│   └── LTSM.ipynb                    # Long Short-Term Memory
│
├── 🤖 Mô hình Machine Learning
│   ├── KNN.ipynb                     # K-Nearest Neighbors
│   ├── KNN2.ipynb
│   ├── SVR.ipynb                     # Support Vector Regression
│   ├── Randomforest.ipynb            # Random Forest
│   ├── BayesianRidge.ipynb           # Bayesian Ridge Regression
│   └── Elastic Net Regression.ipynb  # Elastic Net
│
├── 🧠 Mô hình Deep Learning
│   ├── NBeats1.ipynb                 # N-BEATS Neural Network
│   └── NBeats2.ipynb
│
└── code.ipynb                        # Notebook tổng hợp
```

## 🔧 Các mô hình được sử dụng

| Mô hình               | Mô tả                                        | File                           |
| --------------------- | -------------------------------------------- | ------------------------------ |
| **Linear Regression** | Hồi quy tuyến tính cơ bản                    | `linear_agression_*.ipynb`     |
| **ARIMA**             | Mô hình tự hồi quy tích hợp trung bình trượt | `arima.ipynb`, `ARIMA*.ipynb`  |
| **LSTM**              | Mạng bộ nhớ dài-ngắn hạn                     | `LTSM.ipynb`                   |
| **KNN**               | K láng giềng gần nhất                        | `KNN.ipynb`, `KNN2.ipynb`      |
| **SVR**               | Hồi quy vector hỗ trợ                        | `SVR.ipynb`                    |
| **Random Forest**     | Rừng ngẫu nhiên                              | `Randomforest.ipynb`           |
| **Bayesian Ridge**    | Hồi quy Bayesian Ridge                       | `BayesianRidge.ipynb`          |
| **Elastic Net**       | Kết hợp L1 và L2 regularization              | `Elastic Net Regression.ipynb` |
| **N-BEATS**           | Neural Basis Expansion Analysis              | `NBeats*.ipynb`                |

## 📊 Dữ liệu

- **File**: `World GDP Dataset.csv`
- **Nội dung**: Dữ liệu GDP các quốc gia trên thế giới
- **Định dạng**: CSV

## 🚀 Cài đặt và Chạy

### Yêu cầu hệ thống

```bash
Python 3.8+
```

### Cài đặt thư viện

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install statsmodels tensorflow keras
pip install pytorch-forecasting  # Cho N-BEATS
```

### Chạy ứng dụng GUI

```bash
python app_tkinter.py
```

### Chạy các notebook

Mở các file `.ipynb` bằng Jupyter Notebook hoặc VS Code và chạy từng cell.

## 📈 Quy trình thực hiện

1. **Thu thập dữ liệu**: Sử dụng bộ dữ liệu GDP thế giới
2. **Tiền xử lý**: Làm sạch và chuẩn hóa dữ liệu
3. **Huấn luyện mô hình**: Áp dụng các thuật toán ML/DL
4. **Đánh giá**: So sánh hiệu suất các mô hình (MSE, RMSE, MAE, R²)
5. **Dự báo**: Dự đoán GDP trong tương lai

## 📉 Các chỉ số đánh giá

- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

## 👥 Tác giả

Dự án Hệ Hỗ Trợ Quyết Định - MI2

## 📄 License

MIT License

---

_Dự án được xây dựng cho mục đích học tập và nghiên cứu._
