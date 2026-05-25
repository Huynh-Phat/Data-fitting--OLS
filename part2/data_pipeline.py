import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPipeline:
    def __init__(self, target_col='CO(GT)', imputation_method='knn'):
        self.target_col = target_col
        self.imputation_method = imputation_method
        
        # Khởi tạo Imputer
        if imputation_method == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            self.imputer = None
            
        # Khởi tạo Scaler
        self.scaler = StandardScaler()
        
        # Biến lưu trữ trạng thái sau khi fit
        self.feature_cols = None
        self.lower_bounds = None
        self.upper_bounds = None
        
    def fit(self, X_train):
        """
        Lưu cấu trúc, fit Imputer và Scaler trên tập huấn luyện.
        Tuyệt đối không thay đổi in-place X_train để tránh data leakage.
        """
        # Lưu lại tên các cột nếu là DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_cols = X_train.columns.tolist()
            X_vals = X_train.values
        else:
            self.feature_cols = None
            X_vals = X_train
            
        # 1. Fit Imputer
        if self.imputer:
            self.imputer.fit(X_vals)
            # Biến đổi tạm thời để học Outlier và Scaler
            X_tmp = self.imputer.transform(X_vals)
        else:
            X_tmp = np.copy(X_vals)
            
        # 2. Học các ngưỡng Outlier bằng Winsorization (Percentile 1% và 99%)
        # Việc tính percentile phải thực hiện SAU KHI điền khuyết (nếu không np.nan sẽ lỗi)
        self.lower_bounds = np.percentile(X_tmp, 1, axis=0)
        self.upper_bounds = np.percentile(X_tmp, 99, axis=0)
        
        # Biến đổi cắt Outlier tạm thời để Scaler không bị lệch chuẩn
        X_tmp = np.clip(X_tmp, self.lower_bounds, self.upper_bounds)
        
        # 3. Fit Scaler (Z-score)
        self.scaler.fit(X_tmp)
        
        return self
        
    def transform(self, X):
        """
        Thực hiện chuỗi biến đổi trên X.
        """
        if isinstance(X, pd.DataFrame):
            # Lọc lại đúng thứ tự cột như lúc fit
            if self.feature_cols:
                X_vals = X[self.feature_cols].values
            else:
                X_vals = X.values
        else:
            X_vals = np.copy(X)
            
        # 1. Điền khuyết
        if self.imputer:
            X_transformed = self.imputer.transform(X_vals)
        else:
            X_transformed = np.copy(X_vals)
            
        # 2. Xử lý Outlier (Winsorization - Cắt dải)
        X_transformed = np.clip(X_transformed, self.lower_bounds, self.upper_bounds)
        
        # 3. Chuẩn hóa Z-score
        X_transformed = self.scaler.transform(X_transformed)
        
        # 4. Thêm cột Intercept (toàn số 1)
        ones = np.ones((X_transformed.shape[0], 1))
        X_transformed = np.hstack((ones, X_transformed))
        
        return X_transformed
        
    def fit_transform(self, X_train):
        """
        Gọi liên tiếp fit và transform trên X_train.
        """
        self.fit(X_train)
        return self.transform(X_train)


def load_and_preprocess_raw_data(filepath, target_col='CO(GT)'):
    """
    Load data và áp dụng quy tắc cơ bản của AirQualityUCI.
    """
    try:
        # AirQualityUCI thường dùng ';' làm phân cách và ',' làm dấu thập phân
        df = pd.read_csv(filepath, sep=',', decimal=',')
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {filepath}")
        return None, None
        
    # Loại bỏ các cột/hàng bị trống hoàn toàn do cấu trúc file CSV đôi khi bị dư
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    
    # 1. Giá trị missing là -200 -> Chuyển thành np.nan
    df.replace(-200, np.nan, inplace=True)
    
    # 2. Drop hoàn toàn cột NMHC(GT) nếu có
    if 'NMHC(GT)' in df.columns:
        df.drop(columns=['NMHC(GT)'], inplace=True)
        
    # Loại bỏ cột Date và Time vì đây không phải là dữ liệu numeric trực tiếp cho OLS/Ridge
    if 'Date' in df.columns:
        df.drop(columns=['Date'], inplace=True)
    if 'Time' in df.columns:
        df.drop(columns=['Time'], inplace=True)
        
    # 3. Target CO(GT): Thực hiện Listwise Deletion nếu bị NaN
    if target_col in df.columns:
        df.dropna(subset=[target_col], inplace=True)
    else:
        print(f"Cảnh báo: Không tìm thấy cột Target '{target_col}' trong dataset.")
        return None, None
        
    # Tách X và y
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Ép kiểu dữ liệu về float để đảm bảo KNNImputer chạy tốt
    X = X.astype(float)
    y = y.astype(float)
    
    return X, y

if __name__ == '__main__':
    # Đường dẫn file dữ liệu
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "part2", "data")
    data_path = os.path.join(data_dir, "AirQualityUCI.csv")
    
    # Tạo fake data để test code nếu chưa tải file CSV thực
    if not os.path.exists(data_path):
        print(f"[*] File {data_path} chưa tồn tại.\n[*] Sẽ tự động tạo file giả lập nhỏ để test luồng chạy...")
        os.makedirs(data_dir, exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write("Date;Time;CO(GT);PT08.S1(CO);NMHC(GT);C6H6(GT);PT08.S2(NMHC)\n")
            for i in range(20):
                # Tạo một row bị thiếu Target CO(GT) để test listwise deletion
                co = "2,6" if i != 5 else "-200" 
                # Chèn missing values -200 cho features
                pt08 = "1360" if i % 3 != 0 else "-200"
                f.write(f"10/03/2004;18.00.00;{co};{pt08};-200;11,9;1046\n")
                
    # 1. Load và xử lý Data (Quy tắc đặc thù)
    print("--------------------------------------------------")
    X_raw, y_raw = load_and_preprocess_raw_data(data_path)
    
    if X_raw is not None:
        print(f"Kích thước X sau Listwise Deletion & Drop Cột: {X_raw.shape}")
        
        # 2. Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
        
        # 3. Chạy DataPipeline
        pipeline = DataPipeline(target_col='CO(GT)')
        
        # Fit Transform trên tập Train
        X_train_transformed = pipeline.fit_transform(X_train)
        
        # Chỉ Transform trên tập Test
        X_test_transformed = pipeline.transform(X_test)
        
        print("\nPipeline Hoạt Động Thành Công!")
        print(f"[*] X_train shape sau Pipeline (đã thêm intercept): {X_train_transformed.shape}")
        print(f"[*] X_test shape sau Pipeline (đã thêm intercept): {X_test_transformed.shape}")
        print("\nDòng đầu tiên của X_train_transformed:")
        print(X_train_transformed[0])
