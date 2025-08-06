import tkinter as tk
from tkinter import ttk  # Themed Tkinter widgets for a
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Vẫn có thể dùng matplotlib để vẽ, nhưng nhúng vào Tkinter phức tạp hơn
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # Để nhúng matplotlib
from sklearn.model_selection import train_test_split # Dùng để chia train/test (cần cẩn thận với chuỗi thời gian)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Các độ đo đánh giá
import statsmodels.api as sm
# Import các lớp mô hình và hàm cần thiết của bạn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
darts_available = True
try:
    from pmdarima import auto_arima
    pmdarima_available = True
except ImportError:
    pmdarima_available = False
# import xgboost as xgb # Bỏ comment nếu dùng

# --- PHẦN 1: HÀM TẢI DỮ LIỆU VÀ CÁC HÀM DỰ BÁO ---


def load_and_preprocess_data():
    file_path = 'World GDP Dataset.csv'
    try:
        df_raw = pd.read_csv(file_path, na_values=['0', 0.0, ''])
        df_raw.dropna(how='all', inplace=True)
        df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains("©IMF|World Economic Outlook Database|Data source:", na=False, case=False, regex=True)]
        df_raw.rename(columns={df_raw.columns[0]: 'Country Name'}, inplace=True)
        df_raw.dropna(subset=['Country Name'], inplace=True)
        df_raw.reset_index(drop=True, inplace=True)
        year_columns = df_raw.columns[1:]
        for col in year_columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        df_long_loaded = pd.melt(df_raw, id_vars=['Country Name'], var_name='Year', value_name='GDP')
        df_long_loaded['Year'] = pd.to_numeric(df_long_loaded['Year'], errors='coerce')
        df_long_loaded['GDP'] = pd.to_numeric(df_long_loaded['GDP'], errors='coerce')
        df_long_loaded.dropna(subset=['Year', 'GDP'], inplace=True)
        df_long_loaded.sort_values(by=['Country Name', 'Year'], inplace=True)
        df_long_loaded['Year'] = df_long_loaded['Year'].astype(int)
        df_long_loaded.reset_index(drop=True, inplace=True)
        return df_long_loaded
    except FileNotFoundError:
        messagebox.showerror("Lỗi Dữ liệu", f"Không tìm thấy file: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        messagebox.showerror("Lỗi Dữ liệu", f"Lỗi khi tải dữ liệu: {e}")
        return pd.DataFrame()



def predict_gdp_for_year_lr_tk(country_history_df, target_year, num_lags=3, poly_degree=2):
    # (Logic của hàm forecast_linear_regression_poly từ Streamlit, bỏ st.write)
    # Trả về giá trị dự báo hoặc None nếu lỗi
    print(f"  LR: Đang chuẩn bị và huấn luyện cho năm {target_year}...")
    data = country_history_df.copy()
    feature_cols = []
    for i in range(1, num_lags + 1): data[f'GDP_lag{i}'] = data['GDP'].shift(i); feature_cols.append(f'GDP_lag{i}')
    if poly_degree >= 1: data['Year_Feat'] = data['Year']; feature_cols.append('Year_Feat')
    if poly_degree >= 2: data['Year_Sq_Feat'] = data['Year']**2; feature_cols.append('Year_Sq_Feat')
    
    data_processed = data.dropna(subset=feature_cols + ['GDP']).reset_index(drop=True)
    if len(data_processed) < len(feature_cols) + 5: 
        print(f"LR: Không đủ dữ liệu ({len(data_processed)} điểm) sau khi tạo features.")
        return None

    X_full = data_processed[feature_cols]; y_full = data_processed['GDP']
    model = LinearRegression().fit(X_full, y_full)
    
    gdp_history = pd.Series(country_history_df['GDP'].values, index=country_history_df['Year'])
    last_known_year = country_history_df['Year'].max()
    
    if target_year <= last_known_year:
        target_year_data = data[data['Year'] == target_year]
        if not target_year_data.empty and not target_year_data[feature_cols].isnull().any().any():
            features_for_target = target_year_data[feature_cols]
            return model.predict(features_for_target)[0]
        return gdp_history.get(target_year, None)

    current_lags_vals = []
    for i in range(1, num_lags + 1):
        val = gdp_history.get(last_known_year - i + 1, np.nan)
        if pd.isna(val): print(f"LR: Thiếu lag {i} để dự báo đệ quy."); return None
        current_lags_vals.append(val)
    
    predicted_gdp_target_year = None
    for i_year_pred in range(target_year - last_known_year):
        current_predict_year = last_known_year + 1 + i_year_pred
        features = current_lags_vals.copy()
        if poly_degree >= 1: features.append(current_predict_year)
        if poly_degree >= 2: features.append(current_predict_year**2)
        
        pred_gdp_step = model.predict(np.array(features).reshape(1, -1))[0]
        if current_predict_year == target_year:
            predicted_gdp_target_year = pred_gdp_step
            break
        current_lags_vals.pop(0); current_lags_vals.append(pred_gdp_step)
    return predicted_gdp_target_year


def predict_gdp_for_year_arima_tk(country_history_df, target_year):
    # (Logic của hàm forecast_arima_auto từ Streamlit, bỏ st.write/st.error)
    # Trả về giá trị dự báo hoặc None nếu lỗi
    print(f"  ARIMA: Đang chuẩn bị và huấn luyện cho năm {target_year}...")
    ts_gdp = country_history_df.set_index('Year')['GDP']
    try: ts_gdp.index = pd.to_datetime(ts_gdp.index.astype(str), format='%Y').to_period('A')
    except: pass
    if not pmdarima_available: print("ARIMA: pmdarima chưa cài đặt."); return None
    if len(ts_gdp) < 15: print(f"ARIMA: Dữ liệu quá ngắn ({len(ts_gdp)})."); return None
    try:
        auto_model = auto_arima(ts_gdp, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=False, max_p=3,max_q=3,max_d=2)
        print(f"    ARIMA order: {auto_model.order}")
        model_final = StatsmodelsARIMA(ts_gdp, order=auto_model.order).fit()
        last_known_year = ts_gdp.index[-1].year if isinstance(ts_gdp.index, pd.PeriodIndex) else int(ts_gdp.index[-1])
        if target_year <= last_known_year:
            val = ts_gdp.get(pd.Period(target_year, freq='A') if isinstance(ts_gdp.index, pd.PeriodIndex) else target_year, None)
            if val is not None: print(f"ARIMA: Năm {target_year} là dữ liệu lịch sử. GDP: {val:.3f}")
            return val
        n_steps_to_forecast = target_year - last_known_year
        forecast_series = model_final.forecast(steps=n_steps_to_forecast)
        return forecast_series.iloc[-1]
    except Exception as e: print(f"Lỗi ARIMA: {e}"); return None

def forecast_rf_detrended_future_series(country_history_df, 
                                     target_year, 
                                     num_lags_for_detrended=3,
                                     rf_n_estimators=50,
                                     rf_max_depth=3,
                                     rf_min_samples_split=5,
                                     rf_min_samples_leaf=2,
                                     trend_poly_degree=2):
    """
    Dự báo GDP cho một NĂM MỤC TIÊU CỤ THỂ sử dụng Random Forest trên dữ liệu đã loại bỏ xu hướng.
    Trả về một giá trị float hoặc None.
    """
    model_id_temp = "RF_Detrended_SingleYear"
    print(f"  {model_id_temp}: Bắt đầu cho năm {target_year}...")

    data_hist = country_history_df.copy()
    data_hist.sort_values('Year', inplace=True)
    data_hist.reset_index(drop=True, inplace=True)

    min_data_len = num_lags_for_detrended + 10
    if len(data_hist) < min_data_len:
        print(f"  {model_id_temp}: Không đủ dữ liệu lịch sử ({len(data_hist)}).")
        return None

    # 1. Ước lượng và loại bỏ xu hướng
    data_hist['Time_Index'] = np.arange(len(data_hist))
    trend_feature_cols = ['Time_Index']
    if trend_poly_degree >= 2: data_hist['Time_Index_Sq'] = data_hist['Time_Index'] ** 2; trend_feature_cols.append('Time_Index_Sq')
    if trend_poly_degree >= 3: data_hist['Time_Index_Cub'] = data_hist['Time_Index'] ** 3; trend_feature_cols.append('Time_Index_Cub')
    
    X_trend_hist_features = data_hist[trend_feature_cols]
    y_gdp_hist_original = data_hist['GDP']
    trend_model_final = LinearRegression()
    try:
        trend_model_final.fit(X_trend_hist_features, y_gdp_hist_original)
        data_hist['GDP_Trend'] = trend_model_final.predict(X_trend_hist_features)
        data_hist['GDP_Detrended'] = y_gdp_hist_original - data_hist['GDP_Trend']
    except Exception as e:
        print(f"  {model_id_temp}: Lỗi khi ước lượng xu hướng: {e}"); return None
    
    # 2. Tạo đặc trưng lags cho RF từ GDP_Detrended
    rf_feature_cols = []
    for i in range(1, num_lags_for_detrended + 1):
        col_name = f'GDP_Detrended_lag{i}'
        data_hist[col_name] = data_hist['GDP_Detrended'].shift(i)
        rf_feature_cols.append(col_name)
    
    data_processed_rf_hist = data_hist.dropna(subset=rf_feature_cols + ['GDP_Detrended']).reset_index(drop=True)
    if len(data_processed_rf_hist) < num_lags_for_detrended + 5:
        print(f"  {model_id_temp}: Không đủ dữ liệu sau khi tạo lags detrended."); return None

    X_rf_train_full_hist = data_processed_rf_hist[rf_feature_cols]
    y_rf_train_target_full_hist = data_processed_rf_hist['GDP_Detrended']

    # 3. Huấn luyện RF
    rf_model_final = RandomForestRegressor(
        n_estimators=rf_n_estimators, max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf,
        random_state=42, n_jobs=-1)
    try:
        rf_model_final.fit(X_rf_train_full_hist, y_rf_train_target_full_hist)
    except Exception as e:
        print(f"  {model_id_temp}: Lỗi khi huấn luyện RF: {e}"); return None

    # 4. Dự báo cho target_year
    last_known_year_hist = data_hist['Year'].max()
    last_known_time_index_hist = data_hist['Time_Index'].max()

    if target_year <= last_known_year_hist:
        target_row_processed = data_processed_rf_hist[data_processed_rf_hist['Year'] == target_year]
        if not target_row_processed.empty:
            X_target_rf = target_row_processed[rf_feature_cols]
            predicted_detrended_gdp_target = rf_model_final.predict(X_target_rf)[0]
            trend_component_target = target_row_processed['GDP_Trend'].iloc[0]
            return trend_component_target + predicted_detrended_gdp_target
        elif target_year in data_hist['Year'].values:
             return data_hist[data_hist['Year'] == target_year]['GDP'].iloc[0]
        else: return None

    # Dự báo đệ quy nếu target_year > last_known_year_hist
    current_detrended_lags = []
    for i in range(num_lags_for_detrended, 0, -1): 
        year_for_lag = last_known_year_hist - i + 1
        val_series = data_hist.loc[data_hist['Year'] == year_for_lag, 'GDP_Detrended']
        if val_series.empty or pd.isna(val_series.iloc[0]): return None
        current_detrended_lags.append(val_series.iloc[0])
    
    predicted_gdp_for_target_year = None
    for step in range(target_year - last_known_year_hist):
        current_predict_year = last_known_year_hist + 1 + step
        current_predict_time_index = last_known_time_index_hist + 1 + step
        
        features_rf_step_np = np.array(current_detrended_lags[::-1]).reshape(1, -1)
        predicted_detrended_step = rf_model_final.predict(features_rf_step_np)[0]
        
        trend_features_future_dict = {'Time_Index': [current_predict_time_index]}
        if trend_poly_degree >= 2: trend_features_future_dict['Time_Index_Sq'] = [current_predict_time_index**2]
        if trend_poly_degree >= 3: trend_features_future_dict['Time_Index_Cub'] = [current_predict_time_index**3]
        trend_features_future_df = pd.DataFrame(trend_features_future_dict)[trend_feature_cols]
        predicted_trend_step = trend_model_final.predict(trend_features_future_df)[0]
        
        final_gdp_step = predicted_trend_step + predicted_detrended_step
        
        if current_predict_year == target_year:
            predicted_gdp_for_target_year = final_gdp_step
            break
            
        current_detrended_lags.pop(0) 
        current_detrended_lags.append(predicted_detrended_step) 
            
    return predicted_gdp_for_target_year

def predict_gdp_for_year_svr_tk(country_history_df, 
                                target_year, 
                                num_lags=5, # Dựa trên NUM_LAGS_M6
                                svr_kernel='linear', # Dựa trên kernel='linear'
                                svr_c=1.0,           # Dựa trên C=1.0
                                svr_epsilon=0.1):    # Dựa trên epsilon=0.1
    """
    Dự báo GDP cho một năm mục tiêu sử dụng SVR với các đặc trưng và siêu tham số đã định.
    Dữ liệu X và y sẽ được chuẩn hóa.
    """
    model_id_temp = f"SVR_{svr_kernel}_C{svr_c}_Eps{svr_epsilon}"
    print(f"  {model_id_temp}: Bắt đầu cho năm {target_year} với {num_lags} lags, Year, Year^2...")

    data = country_history_df.copy()
    data.sort_values('Year', inplace=True)
    data.reset_index(drop=True, inplace=True)

    min_data_len = num_lags + 10 
    if len(data) < min_data_len:
        print(f"  {model_id_temp}: Không đủ dữ liệu lịch sử ({len(data)} điểm).")
        return None

    # 1. Tạo đặc trưng
    feature_cols = []
    for i in range(1, num_lags + 1):
        data[f'GDP_lag{i}'] = data['GDP'].shift(i)
        feature_cols.append(f'GDP_lag{i}')
    data['Year_Feat'] = data['Year']
    feature_cols.append('Year_Feat')
    data['Year_Sq_Feat'] = data['Year']**2
    feature_cols.append('Year_Sq_Feat')
    
    data_processed = data.dropna(subset=feature_cols + ['GDP']).reset_index(drop=True)

    if len(data_processed) < len(feature_cols) + 5: # Cần đủ mẫu để huấn luyện
        print(f"  {model_id_temp}: Không đủ dữ liệu sau khi tạo features ({len(data_processed)} dòng).")
        return None

    X_full_raw = data_processed[feature_cols]
    y_full_raw = data_processed['GDP']

    # 2. Chuẩn hóa X và y trên toàn bộ dữ liệu lịch sử có sẵn
    scaler_X_final = StandardScaler()
    X_full_scaled = scaler_X_final.fit_transform(X_full_raw)

    scaler_y_final = StandardScaler()
    y_full_scaled = scaler_y_final.fit_transform(y_full_raw.values.reshape(-1, 1)).flatten()
    
    # 3. Huấn luyện mô hình SVR cuối cùng
    model_final = SVR(kernel=svr_kernel, C=svr_c, epsilon=svr_epsilon)
    try:
        print(f"  {model_id_temp}: Huấn luyện SVR trên {len(X_full_scaled)} mẫu...")
        model_final.fit(X_full_scaled, y_full_scaled) # Huấn luyện trên X_scaled, y_scaled
    except Exception as e:
        print(f"  {model_id_temp}: Lỗi khi huấn luyện SVR: {e}")
        return None

    # 4. Dự báo
    gdp_history_orig_scale = pd.Series(country_history_df['GDP'].values, index=country_history_df['Year'])
    last_known_year = country_history_df['Year'].max()

    # Nếu target_year nằm trong lịch sử và có đủ đặc trưng đã xử lý
    if target_year <= last_known_year:
        target_row_processed_idx = data_processed[data_processed['Year'] == target_year].index
        if not target_row_processed_idx.empty:
            idx_in_X_full = target_row_processed_idx[0] # Lấy index trong X_full_raw
            # Lấy đúng dòng từ X_full_raw để scale và predict
            # (Cần đảm bảo data_processed có index tương ứng với X_full_raw)
            # Cách an toàn hơn là tạo lại features cho target_year
            
            temp_data_for_target_year = data[data['Year'] == target_year].copy()
            if not temp_data_for_target_year.empty and not temp_data_for_target_year[feature_cols].isnull().any().any():
                features_for_target_raw = temp_data_for_target_year[feature_cols]
                features_for_target_scaled = scaler_X_final.transform(features_for_target_raw)
                predicted_y_scaled = model_final.predict(features_for_target_scaled)[0]
                predicted_gdp_orig_scale = scaler_y_final.inverse_transform([[predicted_y_scaled]])[0,0]
                print(f"  {model_id_temp}: Dự báo cho năm lịch sử (có đủ lags) {target_year}.")
                return predicted_gdp_orig_scale
        
        # Nếu là năm lịch sử nhưng không đủ lags cho data_processed, trả về giá trị thực
        if target_year in gdp_history_orig_scale.index:
             actual_gdp = gdp_history_orig_scale[target_year]
             print(f"  {model_id_temp}: Năm {target_year} là dữ liệu lịch sử. Trả về GDP thực tế.")
             return actual_gdp
        else:
            print(f"  {model_id_temp}: Không thể dự báo cho năm lịch sử {target_year} do thiếu dữ liệu.")
            return None


    # Dự báo đệ quy cho tương lai
    if target_year > last_known_year:
        print(f"  {model_id_temp}: Bắt đầu dự báo đệ quy cho các năm sau {last_known_year}...")
        
        # Khởi tạo current_lags_orig_scale từ các giá trị GDP gốc cuối cùng
        current_lags_orig_scale = []
        for i in range(1, num_lags + 1):
            val = gdp_history_orig_scale.get(last_known_year - i + 1, np.nan)
            if pd.isna(val):
                print(f"  {model_id_temp}: Thiếu GDP_lag{i} từ năm {last_known_year - i + 1} để khởi tạo dự báo đệ quy.")
                return None
            current_lags_orig_scale.append(val)
        # current_lags_orig_scale là [GDP(T), GDP(T-1), ..., GDP(T-num_lags+1)]
        
        predicted_gdp_target_year = None

        for step in range(target_year - last_known_year):
            current_predict_year = last_known_year + 1 + step
            
            # Tạo features ở thang đo gốc
            features_list_orig = current_lags_orig_scale.copy() # [lag1, lag2, ..., lagN]
            features_list_orig.append(current_predict_year)      # Year_Feat
            features_list_orig.append(current_predict_year**2)   # Year_Sq_Feat
            
            features_df_orig = pd.DataFrame([features_list_orig], columns=feature_cols)
            
            # Chuẩn hóa features này bằng scaler_X_final
            features_df_scaled = scaler_X_final.transform(features_df_orig)
            
            # Dự đoán y ở thang đo đã chuẩn hóa
            predicted_y_step_scaled = model_final.predict(features_df_scaled)[0]
            
            # Biến đổi ngược y dự đoán về thang đo GDP gốc
            predicted_gdp_step_orig = scaler_y_final.inverse_transform([[predicted_y_step_scaled]])[0,0]
            
            # print(f"    Dự báo cho {current_predict_year}: Scaled_Pred={predicted_y_step_scaled:.3f}, Orig_Pred={predicted_gdp_step_orig:.3f}")

            if current_predict_year == target_year:
                predicted_gdp_target_year = predicted_gdp_step_orig
                break
                
            # Cập nhật current_lags_orig_scale với giá trị GDP gốc vừa dự đoán
            current_lags_orig_scale.pop() # Bỏ lag xa nhất (cuối list)
            current_lags_orig_scale.insert(0, predicted_gdp_step_orig) # Thêm dự đoán mới nhất vào đầu (để nó là lag1)
            
        print(f"  {model_id_temp}: Dự báo đệ quy hoàn tất.")
        return predicted_gdp_target_year
    
    print(f"  {model_id_temp}: Không thể tạo đủ đặc trưng cho năm mục tiêu {target_year}.")
    return None

def predict_gdp_for_year_knn_diff_tk(country_history_df, 
                                     target_year, 
                                     num_diff_lags=3, # Từ NUM_DIFF_LAGS_M7D
                                     n_neighbors=5,   # Từ N_NEIGHBORS_M7D
                                     knn_weights='distance'): # Thêm tham số weights
    """
    Dự báo GDP cho một năm mục tiêu sử dụng KNN trên dữ liệu GDP đã sai phân bậc 1.
    Đặc trưng X (lags của sai phân) sẽ được chuẩn hóa.
    """
    model_id_temp = f"KNN_Diff_k{n_neighbors}"
    print(f"  {model_id_temp}: Bắt đầu cho năm {target_year} với {num_diff_lags} lags của GDP_diff1...")

    data = country_history_df.copy()
    data.sort_values('Year', inplace=True)
    data.reset_index(drop=True, inplace=True)

    min_data_len = num_diff_lags + 1 + 10 # 1 cho diff, num_diff_lags cho lags, 10 cho train/robustness
    if len(data) < min_data_len:
        print(f"  {model_id_temp}: Không đủ dữ liệu lịch sử ({len(data)} điểm, cần ít nhất {min_data_len}).")
        return None

    # 1. Tạo chuỗi sai phân bậc 1
    data['GDP_diff1'] = data['GDP'].diff(1)
    
    # 2. Tạo các lags của chuỗi sai phân GDP_diff1
    feature_cols = []
    for i in range(1, num_diff_lags + 1):
        col_name = f'GDP_diff1_lag{i}'
        data[col_name] = data['GDP_diff1'].shift(i)
        feature_cols.append(col_name)
    
    # 3. Xử lý giá trị NaN và chuẩn bị dữ liệu cho huấn luyện KNN
    # data_processed sẽ chứa các hàng có đủ lags cho GDP_diff1 và GDP_diff1 hiện tại
    # Cũng cần giữ lại GDP(t-1) gốc để tích hợp ngược
    
    # Tạo cột GDP_t_minus_1 trước khi dropna để đảm bảo nó có giá trị cho tất cả các hàng có GDP_diff1
    data['GDP_t_minus_1_orig'] = data['GDP'].shift(1) 
    
    data_processed = data.dropna(subset=feature_cols + ['GDP_diff1', 'GDP_t_minus_1_orig']).reset_index(drop=True)

    if len(data_processed) < num_diff_lags + 5: # Cần đủ mẫu để huấn luyện KNN
        print(f"  {model_id_temp}: Không đủ dữ liệu sau khi tạo features ({len(data_processed)} dòng).")
        return None

    X_full_raw = data_processed[feature_cols]
    y_diff_full_target = data_processed['GDP_diff1'] # Mục tiêu là sai phân hiện tại
    gdp_t_minus_1_for_integration_full = data_processed['GDP_t_minus_1_orig'] # GDP(t-1) gốc

    # 4. Chuẩn hóa Đặc trưng X trên toàn bộ dữ liệu lịch sử có sẵn
    scaler_X_final = StandardScaler()
    X_full_scaled = scaler_X_final.fit_transform(X_full_raw)
    
    # 5. Huấn luyện mô hình KNN cuối cùng
    model_final = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=knn_weights,
        metric='minkowski', 
        p=2
    )
    try:
        print(f"  {model_id_temp}: Huấn luyện KNN trên {len(X_full_scaled)} mẫu (dự đoán GDP_diff1)...")
        model_final.fit(X_full_scaled, y_diff_full_target)
    except Exception as e:
        print(f"  {model_id_temp}: Lỗi khi huấn luyện KNN: {e}")
        return None

    # 6. Dự báo
    last_known_year = country_history_df['Year'].max()
    
    # Nếu target_year nằm trong lịch sử và có đủ đặc trưng đã xử lý
    if target_year <= last_known_year:
        target_row_processed = data_processed[data_processed['Year'] == target_year]
        if not target_row_processed.empty:
            X_target_raw = target_row_processed[feature_cols]
            X_target_scaled = scaler_X_final.transform(X_target_raw) # Chuẩn hóa
            predicted_diff_target = model_final.predict(X_target_scaled)[0]
            
            gdp_t_minus_1_target = target_row_processed['GDP_t_minus_1_orig'].iloc[0]
            final_gdp_prediction = gdp_t_minus_1_target + predicted_diff_target
            print(f"  {model_id_temp}: Dự báo cho năm lịch sử (có đủ lags diff) {target_year}.")
            return final_gdp_prediction
        
        # Nếu là năm lịch sử nhưng không đủ lags cho data_processed, trả về giá trị gốc
        if target_year in country_history_df['Year'].values:
             actual_gdp = country_history_df[country_history_df['Year'] == target_year]['GDP'].iloc[0]
             print(f"  {model_id_temp}: Năm {target_year} là dữ liệu lịch sử. Trả về GDP thực tế.")
             return actual_gdp
        else:
            print(f"  {model_id_temp}: Không thể dự báo cho năm lịch sử {target_year} do thiếu dữ liệu.")
            return None

    # Dự báo đệ quy cho tương lai
    if target_year > last_known_year:
        print(f"  {model_id_temp}: Bắt đầu dự báo đệ quy cho các năm sau {last_known_year}...")
        
        # Khởi tạo current_diff_lags từ các giá trị GDP_diff1 cuối cùng của data_hist
        current_diff_lags = []
        for i in range(num_diff_lags, 0, -1): 
            year_for_lag = last_known_year - i + 1
            val_series = data.loc[data['Year'] == year_for_lag, 'GDP_diff1'] # Lấy từ data đã tính diff1
            if val_series.empty or pd.isna(val_series.iloc[0]):
                print(f"  {model_id_temp}: Thiếu GDP_diff1 cho năm {year_for_lag} để khởi tạo lag {i}.")
                return None
            current_diff_lags.append(val_series.iloc[0])
        # current_diff_lags giờ là [GDP_diff1_lagN, ..., GDP_diff1_lag1] cho năm last_known_year+1
        
        # Lấy giá trị GDP thực tế cuối cùng để bắt đầu tích hợp
        last_actual_gdp_value = data.loc[data['Year'] == last_known_year, 'GDP'].iloc[0]
        
        predicted_gdp_for_target_year = None

        for step in range(target_year - last_known_year):
            current_predict_year = last_known_year + 1 + step
            
            # Tạo X cho KNN (lags của detrended)
            features_knn_step_raw = np.array(current_diff_lags[::-1]).reshape(1, -1) # Đảo ngược cho đúng thứ tự lag1, lag2...
            features_knn_step_scaled = scaler_X_final.transform(features_knn_step_raw) # Chuẩn hóa
            
            # Dự đoán phần diff1 cho current_predict_year
            predicted_diff1_step = model_final.predict(features_knn_step_scaled)[0]
            
            # Tích hợp ngược để có dự đoán GDP
            predicted_gdp_step = last_actual_gdp_value + predicted_diff1_step
            # print(f"    Dự báo cho {current_predict_year}: GDP(t-1)={last_actual_gdp_value:.2f}, Pred_Diff1={predicted_diff1_step:.2f}, Pred_GDP={predicted_gdp_step:.2f}")
            
            if current_predict_year == target_year:
                predicted_gdp_for_target_year = predicted_gdp_step
                break
                
            # Cập nhật current_diff_lags cho vòng lặp tiếp theo
            current_diff_lags.pop(0) 
            current_diff_lags.append(predicted_diff1_step) # Thêm diff vừa dự đoán
            
            # Cập nhật last_actual_gdp_value bằng giá trị GDP vừa dự đoán
            last_actual_gdp_value = predicted_gdp_step
            
        print(f"  {model_id_temp}: Dự báo đệ quy hoàn tất.")
        return predicted_gdp_for_target_year
    
    print(f"  {model_id_temp}: Không thể tạo đủ đặc trưng cho năm mục tiêu {target_year}.")
    return None
def predict_gdp_for_year_elasticnet_tk(country_history_df, 
                                       target_year, 
                                       num_lags=3,         # Từ NUM_LAGS_M8
                                       alpha=0.1,          # Từ ALPHA_M8
                                       l1_ratio=0.5,       # Từ L1_RATIO_M8
                                       include_year_sq=True): # Để kiểm soát việc thêm Year_Sq_Feat
    """
    Dự báo GDP cho một năm mục tiêu sử dụng ElasticNet Regression.
    Đặc trưng X sẽ được chuẩn hóa.
    """
    model_id_temp = f"ElasticNet_Alpha{alpha}_L1r{l1_ratio}"
    print(f"  {model_id_temp}: Bắt đầu cho năm {target_year} với {num_lags} lags, Year, Year^2 (nếu có)...")

    data = country_history_df.copy()
    data.sort_values('Year', inplace=True)
    data.reset_index(drop=True, inplace=True)

    min_data_len = num_lags + 10 
    if len(data) < min_data_len:
        print(f"  {model_id_temp}: Không đủ dữ liệu lịch sử ({len(data)} điểm).")
        return None

    # 1. Tạo đặc trưng
    feature_cols = []
    for i in range(1, num_lags + 1):
        data[f'GDP_lag{i}'] = data['GDP'].shift(i)
        feature_cols.append(f'GDP_lag{i}')
    
    data['Year_Feat'] = data['Year']
    feature_cols.append('Year_Feat')
    
    if include_year_sq:
        data['Year_Sq_Feat'] = data['Year']**2
        feature_cols.append('Year_Sq_Feat')
    
    data_processed = data.dropna(subset=feature_cols + ['GDP']).reset_index(drop=True)

    if len(data_processed) < len(feature_cols) + 5: # Cần đủ mẫu để huấn luyện
        print(f"  {model_id_temp}: Không đủ dữ liệu sau khi tạo features ({len(data_processed)} dòng).")
        return None

    X_full_raw = data_processed[feature_cols]
    y_full_raw = data_processed['GDP']

    # 2. Chuẩn hóa X trên toàn bộ dữ liệu lịch sử có sẵn
    scaler_X_final = StandardScaler()
    X_full_scaled = scaler_X_final.fit_transform(X_full_raw)
    # Lưu ý: ElasticNet thường không yêu cầu chuẩn hóa y mạnh mẽ như SVR hoặc NN,
    # nhưng nếu bạn muốn, bạn có thể thêm scaler_y_final ở đây và inverse_transform sau.
    # Để đơn giản, hàm này sẽ huấn luyện trên y_full_raw.
    
    # 3. Huấn luyện mô hình ElasticNet cuối cùng
    model_final = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42,
        max_iter=10000 # Đảm bảo hội tụ
    )
    try:
        print(f"  {model_id_temp}: Huấn luyện ElasticNet trên {len(X_full_scaled)} mẫu...")
        model_final.fit(X_full_scaled, y_full_raw) # Huấn luyện trên X_scaled, y_gốc
    except Exception as e:
        print(f"  {model_id_temp}: Lỗi khi huấn luyện ElasticNet: {e}")
        return None

    # 4. Dự báo
    gdp_history_orig_scale = pd.Series(country_history_df['GDP'].values, index=country_history_df['Year'])
    last_known_year = country_history_df['Year'].max()

    # Nếu target_year nằm trong lịch sử và có đủ đặc trưng đã xử lý
    if target_year <= last_known_year:
        target_row_data = data[data['Year'] == target_year].copy() # Lấy dòng dữ liệu cho năm mục tiêu
        if not target_row_data.empty:
            # Tạo lại features cho target_year từ dữ liệu gốc (data) để đảm bảo tính nhất quán
            # vì data_processed có thể đã drop các hàng đầu
            # Tuy nhiên, nếu target_year nằm trong data_processed, các lags sẽ có sẵn
            
            # Kiểm tra xem có đủ lags cho target_year không
            can_predict_historical = True
            features_for_target_list = []
            for i in range(1, num_lags + 1):
                lag_val = gdp_history_orig_scale.get(target_year - i, np.nan)
                if pd.isna(lag_val):
                    can_predict_historical = False
                    break
                features_for_target_list.append(lag_val)
            
            if can_predict_historical:
                features_for_target_list.append(target_year) # Year_Feat
                if include_year_sq:
                    features_for_target_list.append(target_year**2) # Year_Sq_Feat
                
                features_for_target_df = pd.DataFrame([features_for_target_list], columns=feature_cols)
                features_for_target_scaled = scaler_X_final.transform(features_for_target_df)
                predicted_gdp_orig_scale = model_final.predict(features_for_target_scaled)[0]
                print(f"  {model_id_temp}: Dự báo cho năm lịch sử (có đủ lags) {target_year}.")
                return predicted_gdp_orig_scale
        
        # Nếu là năm lịch sử nhưng không đủ lags, trả về giá trị thực
        if target_year in gdp_history_orig_scale.index:
             actual_gdp = gdp_history_orig_scale[target_year]
             print(f"  {model_id_temp}: Năm {target_year} là dữ liệu lịch sử. Trả về GDP thực tế.")
             return actual_gdp
        else:
            print(f"  {model_id_temp}: Không thể dự báo cho năm lịch sử {target_year} do thiếu dữ liệu.")
            return None


    # Dự báo đệ quy cho tương lai
    if target_year > last_known_year:
        print(f"  {model_id_temp}: Bắt đầu dự báo đệ quy cho các năm sau {last_known_year}...")
        
        current_lags_orig_scale = []
        for i in range(1, num_lags + 1):
            val = gdp_history_orig_scale.get(last_known_year - i + 1, np.nan)
            if pd.isna(val):
                print(f"  {model_id_temp}: Thiếu GDP_lag{i} từ năm {last_known_year - i + 1} để khởi tạo dự báo đệ quy.")
                return None
            current_lags_orig_scale.append(val)
        # current_lags_orig_scale là [GDP(T), GDP(T-1), ..., GDP(T-num_lags+1)]
        
        predicted_gdp_target_year = None

        for step in range(target_year - last_known_year):
            current_predict_year = last_known_year + 1 + step
            
            features_list_orig = current_lags_orig_scale.copy() # [lag1, lag2, ..., lagN]
            features_list_orig.append(current_predict_year)      # Year_Feat
            if include_year_sq:
                features_list_orig.append(current_predict_year**2)   # Year_Sq_Feat
            
            features_df_orig = pd.DataFrame([features_list_orig], columns=feature_cols)
            features_df_scaled = scaler_X_final.transform(features_df_orig) # Chuẩn hóa
            
            predicted_gdp_step_orig = model_final.predict(features_df_scaled)[0]
            # print(f"    Dự báo cho {current_predict_year}: Pred_GDP={predicted_gdp_step_orig:.3f}")

            if current_predict_year == target_year:
                predicted_gdp_target_year = predicted_gdp_step_orig
                break
                
            current_lags_orig_scale.pop() 
            current_lags_orig_scale.insert(0, predicted_gdp_step_orig) 
            
        print(f"  {model_id_temp}: Dự báo đệ quy hoàn tất.")
        return predicted_gdp_target_year
    
    print(f"  {model_id_temp}: Không thể tạo đủ đặc trưng cho năm mục tiêu {target_year}.")
    return None

def predict_gdp_for_year_bayesian_ridge_tk(country_history_df, 
                                           target_year, 
                                           num_lags=3,         # Từ NUM_LAGS_M8
                                           include_year_sq=True, # Để kiểm soát việc thêm Year_Sq_Feat
                                           br_max_iter=300,    # Từ max_iter trong code của bạn
                                           br_tol=1e-3):       # Từ tol trong code của bạn
    """
    Dự báo GDP cho một năm mục tiêu sử dụng Bayesian Ridge Regression.
    Đặc trưng X sẽ được chuẩn hóa.
    """
    model_id_temp = f"BayesianRidge_Lags{num_lags}_YearSq{include_year_sq}"
    print(f"  {model_id_temp}: Bắt đầu cho năm {target_year}...")

    data = country_history_df.copy()
    data.sort_values('Year', inplace=True)
    data.reset_index(drop=True, inplace=True)

    min_data_len = num_lags + 10 
    if len(data) < min_data_len:
        print(f"  {model_id_temp}: Không đủ dữ liệu lịch sử ({len(data)} điểm).")
        return None

    # 1. Tạo đặc trưng
    feature_cols = []
    for i in range(1, num_lags + 1):
        data[f'GDP_lag{i}'] = data['GDP'].shift(i)
        feature_cols.append(f'GDP_lag{i}')
    
    data['Year_Feat'] = data['Year']
    feature_cols.append('Year_Feat')
    
    if include_year_sq:
        data['Year_Sq_Feat'] = data['Year']**2
        feature_cols.append('Year_Sq_Feat')
    
    data_processed = data.dropna(subset=feature_cols + ['GDP']).reset_index(drop=True)

    if len(data_processed) < len(feature_cols) + 5: # Cần đủ mẫu để huấn luyện
        print(f"  {model_id_temp}: Không đủ dữ liệu sau khi tạo features ({len(data_processed)} dòng).")
        return None

    X_full_raw = data_processed[feature_cols]
    y_full_raw = data_processed['GDP'] # Sử dụng y ở thang đo gốc

    # 2. Chuẩn hóa X trên toàn bộ dữ liệu lịch sử có sẵn
    scaler_X_final = StandardScaler()
    X_full_scaled = scaler_X_final.fit_transform(X_full_raw)
    
    # 3. Huấn luyện mô hình BayesianRidge cuối cùng
    model_final = BayesianRidge(
        max_iter=br_max_iter,
        tol=br_tol,
        compute_score=True # Giữ nguyên như trong code của bạn
        # Các tham số alpha_init, lambda_init, alpha_1, alpha_2, lambda_1, lambda_2
        # có thể để mặc định để mô hình tự ước lượng.
    )
    try:
        print(f"  {model_id_temp}: Huấn luyện Bayesian Ridge trên {len(X_full_scaled)} mẫu...")
        model_final.fit(X_full_scaled, y_full_raw) # Huấn luyện trên X_scaled, y_gốc
        print(f"    Alpha (ước lượng): {model_final.alpha_:.6f}, Lambda (ước lượng): {model_final.lambda_:.6f}")
    except Exception as e:
        print(f"  {model_id_temp}: Lỗi khi huấn luyện Bayesian Ridge: {e}")
        return None

    # 4. Dự báo
    gdp_history_orig_scale = pd.Series(country_history_df['GDP'].values, index=country_history_df['Year'])
    last_known_year = country_history_df['Year'].max()

    # Nếu target_year nằm trong lịch sử và có đủ đặc trưng đã xử lý
    if target_year <= last_known_year:
        # Tạo lại features cho target_year từ dữ liệu gốc 'data' (trước khi dropna)
        # để đảm bảo có thể lấy được các lags nếu target_year nằm ở đầu data_processed
        target_year_data_row = data[data['Year'] == target_year].copy() # Lấy dòng dữ liệu cho năm mục tiêu
        
        if not target_year_data_row.empty:
            # Kiểm tra xem có đủ lags cho target_year không
            can_predict_historical = True
            features_for_target_list = []
            for i in range(1, num_lags + 1):
                # Lấy lag từ gdp_history_orig_scale để đảm bảo là giá trị gốc
                lag_val = gdp_history_orig_scale.get(target_year - i, np.nan)
                if pd.isna(lag_val):
                    can_predict_historical = False
                    break
                features_for_target_list.append(lag_val)
            
            if can_predict_historical:
                features_for_target_list.append(target_year) # Year_Feat
                if include_year_sq:
                    features_for_target_list.append(target_year**2) # Year_Sq_Feat
                
                if len(features_for_target_list) == len(feature_cols):
                    features_for_target_df = pd.DataFrame([features_for_target_list], columns=feature_cols)
                    features_for_target_scaled = scaler_X_final.transform(features_for_target_df)
                    predicted_gdp_orig_scale = model_final.predict(features_for_target_scaled)[0]
                    print(f"  {model_id_temp}: Dự báo cho năm lịch sử (có đủ lags) {target_year}.")
                    return predicted_gdp_orig_scale
        
        # Nếu là năm lịch sử nhưng không đủ lags cho data_processed, trả về giá trị thực
        if target_year in gdp_history_orig_scale.index:
             actual_gdp = gdp_history_orig_scale[target_year]
             print(f"  {model_id_temp}: Năm {target_year} là dữ liệu lịch sử. Trả về GDP thực tế.")
             return actual_gdp
        else:
            print(f"  {model_id_temp}: Không thể dự báo cho năm lịch sử {target_year} do thiếu dữ liệu.")
            return None


    # Dự báo đệ quy cho tương lai
    if target_year > last_known_year:
        print(f"  {model_id_temp}: Bắt đầu dự báo đệ quy cho các năm sau {last_known_year}...")
        
        current_lags_orig_scale = []
        for i in range(1, num_lags + 1):
            val = gdp_history_orig_scale.get(last_known_year - i + 1, np.nan)
            if pd.isna(val):
                print(f"  {model_id_temp}: Thiếu GDP_lag{i} từ năm {last_known_year - i + 1} để khởi tạo dự báo đệ quy.")
                return None
            current_lags_orig_scale.append(val)
        # current_lags_orig_scale là [GDP(T), GDP(T-1), ..., GDP(T-num_lags+1)]
        
        predicted_gdp_target_year = None

        for step in range(target_year - last_known_year):
            current_predict_year = last_known_year + 1 + step
            
            features_list_orig = current_lags_orig_scale.copy() # [lag1, lag2, ..., lagN]
            features_list_orig.append(current_predict_year)      # Year_Feat
            if include_year_sq:
                features_list_orig.append(current_predict_year**2)   # Year_Sq_Feat
            
            features_df_orig = pd.DataFrame([features_list_orig], columns=feature_cols)
            features_df_scaled = scaler_X_final.transform(features_df_orig) # Chuẩn hóa
            
            predicted_gdp_step_orig = model_final.predict(features_df_scaled)[0]
            # print(f"    Dự báo cho {current_predict_year}: Pred_GDP={predicted_gdp_step_orig:.3f}")

            if current_predict_year == target_year:
                predicted_gdp_target_year = predicted_gdp_step_orig
                break
                
            current_lags_orig_scale.pop() 
            current_lags_orig_scale.insert(0, predicted_gdp_step_orig) 
            
        print(f"  {model_id_temp}: Dự báo đệ quy hoàn tất.")
        return predicted_gdp_target_year
    
    print(f"  {model_id_temp}: Không thể tạo đủ đặc trưng cho năm mục tiêu {target_year}.")
    return None

def forecast_nbeats_gdp_for_year_tk(country_history_df, 
                                     target_year, # Năm cụ thể cần dự báo hoặc xem lịch sử
                                     # Siêu tham số N-BEATS (nên lấy từ kết quả tối ưu của bạn)
                                     input_chunk_length=10, 
                                     output_chunk_length=1, # QUAN TRỌNG: để là 1 cho dự báo đệ quy từng bước
                                     num_stacks=3,          
                                     num_blocks=2,          
                                     num_layers=2,          
                                     layer_widths=128,       
                                     n_epochs=57,           
                                     batch_size=8,          
                                     learning_rate=0.0007591104805282694,    
                                     model_id_prefix="NBEATS_App_TargetYear"):
    """
    Lấy giá trị GDP thực tế nếu target_year nằm trong lịch sử,
    hoặc dự báo GDP cho một NĂM MỤC TIÊU CỤ THỂ trong tương lai sử dụng N-BEATS.
    Hàm này sẽ huấn luyện mô hình trên toàn bộ country_history_df nếu cần dự báo tương lai.
    Trả về một giá trị float GDP, hoặc None nếu lỗi.
    """
    if not darts_available:
        print(f"  {model_id_prefix}: Thư viện Darts không khả dụng. Không thể chạy N-BEATS.")
        return None

    country_name = country_history_df['Country Name'].iloc[0] if not country_history_df.empty else "Unknown"
    print(f"  {model_id_prefix}: Xử lý cho năm {target_year} cho {country_name}...")

    data_hist_for_model = country_history_df.copy()
    data_hist_for_model.sort_values('Year', inplace=True)
    data_hist_for_model.reset_index(drop=True, inplace=True)

    last_known_year_hist = data_hist_for_model['Year'].max()
    first_known_year_hist = data_hist_for_model['Year'].min()

    # --- Trường hợp 1: target_year nằm trong khoảng dữ liệu lịch sử ---
    if target_year >= first_known_year_hist and target_year <= last_known_year_hist:
        actual_value_series = data_hist_for_model[data_hist_for_model['Year'] == target_year]['GDP']
        if not actual_value_series.empty:
            actual_gdp = actual_value_series.iloc[0]
            print(f"  {model_id_prefix}: Năm {target_year} là dữ liệu lịch sử. GDP thực tế: {actual_gdp:.3f}")
            return actual_gdp
        else:
            # Điều này ít khi xảy ra nếu target_year nằm trong min/max của cột Year
            print(f"  {model_id_prefix}: Không tìm thấy dữ liệu lịch sử cho năm {target_year} (dù nằm trong khoảng).")
            return None # Hoặc bạn có thể thử dự báo in-sample nếu muốn

    # --- Trường hợp 2: target_year nằm trong tương lai ---
    if target_year > last_known_year_hist:
        print(f"  {model_id_prefix}: Năm {target_year} là năm tương lai. Bắt đầu quá trình dự báo...")
        # Kiểm tra dữ liệu đầu vào đủ dài để huấn luyện
        min_data_len_for_fit = input_chunk_length + output_chunk_length
        if len(data_hist_for_model) < min_data_len_for_fit:
            print(f"  {model_id_prefix}: Không đủ dữ liệu lịch sử ({len(data_hist_for_model)} điểm, cần ít nhất {min_data_len_for_fit}) để huấn luyện N-BEATS.")
            return None

        # 1. Chuẩn bị TimeSeries của Darts và Scaler
        ts_gdp_darts_full = None
        scaler_darts_local = None 
        ts_gdp_scaled_full_darts = None
        try:
            data_hist_for_model['TimeCol'] = pd.to_datetime(data_hist_for_model['Year'].astype(str) + '-01-01')
            ts_gdp_darts_full = TimeSeries.from_dataframe(data_hist_for_model, 'TimeCol', 'GDP', freq='AS-JAN')
            scaler_darts_local = Scaler() 
            ts_gdp_scaled_full_darts = scaler_darts_local.fit_transform(ts_gdp_darts_full)
            print(f"  {model_id_prefix}: Dữ liệu lịch sử đã được chuẩn bị và chuẩn hóa ({len(ts_gdp_scaled_full_darts)} điểm).")
        except Exception as e:
            print(f"  {model_id_prefix}: Lỗi khi chuẩn bị TimeSeries hoặc Scaler: {e}")
            return None 

        # 2. Khởi tạo và Huấn luyện Mô hình N-BEATS
        model_nbeats_final = NBEATSModel(
            input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length,
            num_stacks=num_stacks, num_blocks=num_blocks, num_layers=num_layers, layer_widths=layer_widths,
            n_epochs=n_epochs, batch_size=batch_size, optimizer_kwargs={'lr': learning_rate},
            random_state=42, model_name=f"{model_id_prefix}_{country_name}_target_{target_year}", force_reset=True,
        )
        try:
            print(f"  {model_id_prefix}: Huấn luyện N-BEATS trên toàn bộ lịch sử...")
            model_nbeats_final.fit(ts_gdp_scaled_full_darts, verbose=False) 
            print(f"  {model_id_prefix}: Huấn luyện N-BEATS hoàn tất.")
        except Exception as e:
            print(f"  {model_id_prefix}: Lỗi khi huấn luyện N-BEATS: {e}")
            return None

        # 3. Dự báo cho target_year
        try:
            n_steps_to_forecast = target_year - last_known_year_hist
            print(f"  {model_id_prefix}: Tạo dự báo cho {n_steps_to_forecast} bước tới (đến năm {target_year})...")
            
            forecast_nbeats_future_scaled = model_nbeats_final.predict(n=n_steps_to_forecast, series=ts_gdp_scaled_full_darts)
            
            if scaler_darts_local is None:
                print(f"  {model_id_prefix}: Lỗi - Scaler chưa được khởi tạo để biến đổi ngược.")
                return None
                
            forecast_nbeats_future_orig = scaler_darts_local.inverse_transform(forecast_nbeats_future_scaled)
            predicted_gdp_for_target = forecast_nbeats_future_orig.pd_series().iloc[-1]
            
            print(f"  {model_id_prefix}: Dự báo cho năm {target_year} hoàn tất.")
            return predicted_gdp_for_target
            
        except Exception as e:
            print(f"  {model_id_prefix}: Lỗi khi dự báo tương lai với N-BEATS: {e}")
            import traceback
            traceback.print_exc() 
            return None
    
    # Trường hợp target_year < first_known_year_hist (không hợp lệ)
    print(f"  {model_id_prefix}: Năm mục tiêu {target_year} nằm trước khoảng dữ liệu lịch sử ({first_known_year_hist}-{last_known_year_hist}).")
    return None
# --- PHẦN 2: XÂY DỰNG GIAO DIỆN TKINTER ---

class GDPForecastApp:
    def __init__(self, master):
        self.master = master
        master.title("Dự báo GDP ")
        master.geometry("600x350") # Kích thước cửa sổ

        self.df_long = load_and_preprocess_data()
        if self.df_long.empty:
            master.destroy() # Đóng app nếu không tải được dữ liệu
            return

        # --- Tạo các Widget ---
        ttk.Label(master, text="Chọn Quốc gia:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.country_var = tk.StringVar()
        self.country_combo = ttk.Combobox(master, textvariable=self.country_var, width=30)
        self.country_combo['values'] = sorted(self.df_long['Country Name'].unique().tolist())
        self.country_combo.current(self.country_combo['values'].index('Vietnam') if 'Vietnam' in self.country_combo['values'] else 0)
        self.country_combo.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.country_combo.bind("<<ComboboxSelected>>", self.update_year_options) # Cập nhật năm khi chọn quốc gia

        ttk.Label(master, text="Chọn Mô hình:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.model_var = tk.StringVar()
        self.model_options_map_tk = {
            "Hồi quy Tuyến tính": "LR_Poly",
            "ARIMA ": "ARIMA_Auto",
            "Random Forest ": "RF_Detrended",
            "SVR ": "SVR_Linear",
            "KNN": "KNN_Diff",
            "ElasticNet": "ElasticNet",
            "Bayesian Ridge":"Bayesian Ridge",
            "NBEATS":"NBEATS"
            # Thêm các mô hình khác
        }
        self.model_combo = ttk.Combobox(master, textvariable=self.model_var, values=list(self.model_options_map_tk.keys()), width=30)
        self.model_combo.current(0)
        self.model_combo.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        ttk.Label(master, text="Chọn Năm Dự báo:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.year_var = tk.IntVar()
        self.year_spinbox = ttk.Spinbox(master, textvariable=self.year_var, width=10) # Dùng Spinbox thay slider
        self.year_spinbox.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.update_year_options() # Gọi lần đầu để thiết lập năm

        self.forecast_button = ttk.Button(master, text="Dự báo GDP", command=self.perform_forecast)
        self.forecast_button.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Separator(master, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky='ew', pady=5)

        ttk.Label(master, text="Kết quả Dự báo:", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, pady=5)
        self.result_label_text = tk.StringVar(value="GDP Dự báo (Tỷ USD): --")
        self.result_label = ttk.Label(master, textvariable=self.result_label_text, font=("Arial", 14))
        self.result_label.grid(row=6, column=0, columnspan=2, pady=10)
        
        # (Tùy chọn) Khu vực để vẽ biểu đồ (phức tạp hơn)
        # self.fig_frame = ttk.Frame(master)
        # self.fig_frame.grid(row=7, column=0, columnspan=2, pady=5)

        # Cấu hình grid
        master.columnconfigure(1, weight=1)


    def update_year_options(self, event=None):
        selected_c = self.country_var.get()
        if not self.df_long.empty and selected_c:
            country_hist_df = self.df_long[self.df_long['Country Name'] == selected_c]
            if not country_hist_df.empty:
                current_max_year = country_hist_df['Year'].max()
                min_pred_year = int(current_max_year) + 1
                max_pred_year = int(current_max_year) + 10
                self.year_spinbox.config(from_=min_pred_year, to=max_pred_year)
                self.year_var.set(min_pred_year) # Đặt giá trị mặc định
            else: # Nếu quốc gia không có dữ liệu (ít khả năng xảy ra sau khi lọc)
                self.year_spinbox.config(from_=2024, to=2033)
                self.year_var.set(2024)
        else: # Nếu df_long rỗng
            self.year_spinbox.config(from_=2024, to=2033)
            self.year_var.set(2024)


    def perform_forecast(self):
        selected_c = self.country_var.get()
        selected_m_display = self.model_var.get()
        selected_y = self.year_var.get()

        if not selected_c or not selected_m_display or not selected_y:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn đầy đủ Quốc gia, Mô hình và Năm.")
            return

        self.result_label_text.set(f"Đang xử lý cho {selected_c}, năm {selected_y}...")
        self.master.update_idletasks() # Cập nhật giao diện

        country_history_df = self.df_long[self.df_long['Country Name'] == selected_c]
        if country_history_df.empty or len(country_history_df) < 10:
            messagebox.showerror("Lỗi Dữ liệu", f"Không đủ dữ liệu lịch sử cho {selected_c}.")
            self.result_label_text.set("GDP Dự báo (Tỷ USD): Lỗi dữ liệu")
            return

        predicted_gdp = None
        model_id_internal = self.model_options_map_tk[selected_m_display]

        # --- Gọi hàm dự báo tương ứng ---
        # (Đây là nơi bạn sẽ chạy mô hình. Với các mô hình chạy lâu, bạn nên dùng threading)
        try:
            if model_id_internal == "LR_Poly":
                predicted_gdp = predict_gdp_for_year_lr_tk(country_history_df, target_year=selected_y, num_lags=3, poly_degree=2)
            elif model_id_internal == "ARIMA_Auto":
                predicted_gdp = predict_gdp_for_year_arima_tk(country_history_df, target_year=selected_y)
            elif model_id_internal == "RF_Detrended":
                predicted_gdp = forecast_rf_detrended_future_series(country_history_df, 
                                     target_year=selected_y, 
                                     num_lags_for_detrended=3,
                                     rf_n_estimators=50,
                                     rf_max_depth=3,
                                     rf_min_samples_split=5,
                                     rf_min_samples_leaf=2)
            elif model_id_internal == "SVR_Linear":
                predicted_gdp = predict_gdp_for_year_svr_tk(country_history_df, 
                                target_year=selected_y, 
                                num_lags=5, # Dựa trên NUM_LAGS_M6
                                svr_kernel='linear', # Dựa trên kernel='linear'
                                svr_c=1.0,           # Dựa trên C=1.0
                                svr_epsilon=0.1)  
            elif model_id_internal == "KNN_Diff":
                predicted_gdp = predict_gdp_for_year_knn_diff_tk(country_history_df, 
                                     target_year=selected_y, 
                                     num_diff_lags=3, # Từ NUM_DIFF_LAGS_M7D
                                     n_neighbors=5,   # Từ N_NEIGHBORS_M7D
                                     knn_weights='distance') # Thêm tham số weights  
            elif model_id_internal == "ElasticNet":
                predicted_gdp = predict_gdp_for_year_elasticnet_tk(country_history_df, 
                                       target_year=selected_y, 
                                       num_lags=3,         # Từ NUM_LAGS_M8
                                       alpha=0.1,          # Từ ALPHA_M8
                                       l1_ratio=0.5,       # Từ L1_RATIO_M8
                                       include_year_sq=True) 
            elif model_id_internal == "Bayesian Ridge":
                predicted_gdp = predict_gdp_for_year_bayesian_ridge_tk(country_history_df, 
                                           target_year=selected_y, 
                                           num_lags=3,         # Từ NUM_LAGS_M8
                                           include_year_sq=True, # Để kiểm soát việc thêm Year_Sq_Feat
                                           br_max_iter=300,    # Từ max_iter trong code của bạn
                                           br_tol=1e-3)     
            elif model_id_internal == "NBEATS": # ID bạn đặt cho N-BEATS
                if darts_available: 
                    selected_y = self.year_var.get() # Lấy năm mục tiêu từ widget
                    
                    predicted_gdp = forecast_nbeats_gdp_for_year_tk(country_history_df, 
                                     target_year=selected_y, # Năm cụ thể cần dự báo hoặc xem lịch sử
                                     # Siêu tham số N-BEATS (nên lấy từ kết quả tối ưu của bạn)
                                     input_chunk_length=10, 
                                     output_chunk_length=1, # QUAN TRỌNG: để là 1 cho dự báo đệ quy từng bước
                                     num_stacks=3,          
                                     num_blocks=2,          
                                     num_layers=2,          
                                     layer_widths=128,       
                                     n_epochs=57,           
                                     batch_size=8,          
                                     learning_rate=0.0007591104805282694,    
                                     model_id_prefix="NBEATS_App_TargetYear")
                    if predicted_gdp is not None:
                        # Kiểm tra xem có phải là giá trị lịch sử hay dự báo
                        if selected_y <= country_history_df['Year'].max():
                            output_message = f"GDP Lịch sử năm {selected_y} (N-BEATS):\n{predicted_gdp:.3f} Tỷ USD"
                        else:
                            output_message = f"GDP Dự báo cho năm {selected_y} (N-BEATS):\n{predicted_gdp:.3f} Tỷ USD"
                    else:
                        output_message = f"Không thể lấy/dự báo GDP cho năm {selected_y} (N-BEATS)."
                else:
                    output_message = "Lỗi: Thư viện Darts chưa được cài đặt để chạy N-BEATS."        
            # --- THÊM CÁC ELIF CHO CÁC MÔ HÌNH KHÁC ---
            # elif model_id_internal == "XGBoost_LagsYear":
            #     predicted_gdp = predict_gdp_for_year_xgboost_tk(country_history_df, target_year=selected_y, num_lags=5)
            else:
                messagebox.showwarning("Chưa hỗ trợ", f"Mô hình '{selected_m_display}' chưa được triển khai.")
                self.result_label_text.set("GDP Dự báo (Tỷ USD): Mô hình chưa hỗ trợ")
                return

            if predicted_gdp is not None:
                self.result_label_text.set(f"GDP Dự báo ({selected_y}): {predicted_gdp:.3f} Tỷ USD")
                # (Tùy chọn) Vẽ biểu đồ đơn giản và hiển thị (phức tạp hơn để nhúng vào Tkinter)
                # self.plot_forecast_tk(country_history_df, selected_y, predicted_gdp)
            else:
                self.result_label_text.set(f"GDP Dự báo ({selected_y}): Không thể tính toán")
        except Exception as e:
            messagebox.showerror("Lỗi Mô hình", f"Lỗi khi chạy mô hình {selected_m_display}: {e}")
            self.result_label_text.set("GDP Dự báo (Tỷ USD): Lỗi mô hình")

    


# --- PHẦN 3: CHẠY ỨNG DỤNG TKINTER ---
if __name__ == "__main__":
    root = tk.Tk()
    # Kiểm tra pmdarima trước khi khởi tạo app nếu ARIMA là lựa chọn
    if not pmdarima_available and "ARIMA (Tự động)" in GDPForecastApp(root).model_options_map_tk.values(): # Khởi tạo tạm để check
         response = messagebox.askokcancel("Thiếu thư viện", 
                                       "Thư viện 'pmdarima' cần cho mô hình ARIMA (Tự động) chưa được cài đặt.\n"
                                       "Bạn có muốn tiếp tục mà không có chức năng này không?\n"
                                       "(Cài đặt bằng: pip install pmdarima)")
         if not response:
             root.destroy()
             exit() # Thoát nếu người dùng không muốn tiếp tục

    # Tạo đối tượng ứng dụng sau khi kiểm tra (hoặc nếu không có ARIMA)
    if 'root' in locals() and root.winfo_exists(): # Kiểm tra root còn tồn tại không
        app = GDPForecastApp(root)
        root.mainloop()