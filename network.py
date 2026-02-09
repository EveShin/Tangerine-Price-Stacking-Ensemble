"""
감귤 가격 예측 프로젝트
3개 모델 앙상블 network 코드 (LightGBM Enhanced + Random Forest + XGBoost)
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
import pickle
import time

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

REPORT = []

def log(message):
    REPORT.append(message)
    print(message)

# 전처리된 데이터 로드
def load_data(data_dir):
    log("=" * 70)
    log("데이터 로드 - 앙상블 분석")
    log("=" * 70)

    train = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    val = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))

    log(f"\nTrain: {train.shape}")
    log(f"Val:   {val.shape}")
    log(f"Test:  {test.shape}")

    return train, val, test

# 향상된 피처 엔지니어링 - 추가 피처 생성
def create_enhanced_features(df):
    df = df.copy()

    # 1. 소비 관련 피처
    if 'estimated_sales' in df.columns:
        df['consumption_yoy'] = df['estimated_sales'].pct_change(12)  # 전년 대비 증감률
        df['consumption_growth'] = df['estimated_sales'].pct_change(1)  # 전월 대비 증감률
        if 'total_supply' in df.columns:
            df['consumption_per_supply'] = df['estimated_sales'] / (df['total_supply'] + 1)  # 소비/공급 비율
        df['consumption_roll4w'] = df['estimated_sales'].rolling(window=4, min_periods=1).mean()  # 4주 이동평균
        consumption_mean = df['estimated_sales'].mean()
        df['consumption_deviation'] = (df['estimated_sales'] - consumption_mean) / consumption_mean  # 평균 대비 편차

    # 2. 수출/무역 관련 피처
    if 'export_amount' in df.columns:
        df['export_yoy'] = df['export_amount'].pct_change(12)  # 전년 대비 증감률
        df['export_growth'] = df['export_amount'].pct_change(1)  # 전월 대비 증감률
        if 'total_production' in df.columns:
            df['export_ratio'] = df['export_amount'] / (df['total_production'] + 1)  # 수출/생산 비율
        if 'total_supply' in df.columns:
            df['domestic_supply_adjusted'] = df['total_supply'] - df['export_amount'].fillna(0)  # 국내 공급량

    # 3. 상호작용 피처
    if 'total_supply' in df.columns and 'is_peak_season' in df.columns:
        df['supply_x_peak'] = df['total_supply'] * df['is_peak_season']  # 공급 × 성수기
    if 'total_supply' in df.columns and 'is_off_season' in df.columns:
        df['supply_x_off'] = df['total_supply'] * df['is_off_season']  # 공급 × 비수기
    if 'temp_avg' in df.columns and 'rainfall' in df.columns:
        df['temp_x_rain'] = df['temp_avg'] * df['rainfall']  # 기온 × 강수량
    if 'total_supply' in df.columns and 'export_amount' in df.columns:
        df['supply_x_export'] = df['total_supply'] * df['export_amount'].fillna(0)  # 공급 × 수출

    # 4. 가격 변동성
    if 'price_per_kg_avg' in df.columns:
        df['price_range_4w'] = (
                df['price_per_kg_avg'].rolling(window=4, min_periods=1).max() -
                df['price_per_kg_avg'].rolling(window=4, min_periods=1).min()
        )  # 4주 가격 범위
        df['price_cv_4w'] = (
                df['price_per_kg_avg'].rolling(window=4, min_periods=1).std() /
                df['price_per_kg_avg'].rolling(window=4, min_periods=1).mean()
        )  # 4주 변동계수

    # 5. 공급 변동성
    if 'total_supply' in df.columns:
        df['supply_range_4w'] = (
                df['total_supply'].rolling(window=4, min_periods=1).max() -
                df['total_supply'].rolling(window=4, min_periods=1).min()
        )  # 4주 공급 범위

    # 6. 계절성 강화
    if 'month' in df.columns:
        df['is_month_10'] = (df['month'] == 10).astype(int)  # 10월 (성수기)
        df['is_month_11'] = (df['month'] == 11).astype(int)  # 11월 (성수기)
        df['is_month_12'] = (df['month'] == 12).astype(int)  # 12월 (성수기)
        df['is_month_6'] = (df['month'] == 6).astype(int)  # 6월 (비수기)
        df['is_month_7'] = (df['month'] == 7).astype(int)  # 7월 (비수기)

    return df

# 피처 선택
def select_features(train, val, test):
    # 피처 엔지니어링 적용
    train = create_enhanced_features(train)
    val = create_enhanced_features(val)
    test = create_enhanced_features(test)

    # 제거할 피처 (다중공선성, 불필요한 피처)
    remove_features = [
        'date', 'year', 'week_of_year', 'day_of_year',
        'temp_min', 'temp_max',
        'supply_seoul', 'supply_nonghyup', 'supply_jungang',
        'supply_donghwa', 'supply_hanguk',
        'export_weight',
        'production_noji', 'production_house', 'production_winter',
        'production_noji_late', 'production_house_late',
    ]

    # 피처 목록 생성
    all_features = [col for col in train.columns if col != 'date']
    target = 'price_per_kg_avg'
    features = [f for f in all_features if f not in remove_features and f != target]

    # X, y 분리
    X_train = train[features]
    y_train = train[target]
    X_val = val[features]
    y_val = val[target]
    X_test = test[features]
    y_test = test[target]

    # 결측치 제거
    train_mask = y_train.notna()
    val_mask = y_val.notna()
    test_mask = y_test.notna()

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    return X_train, X_val, X_test, y_train, y_val, y_test, features, train, val, test

# 성능지표
def calculate_metrics(y_true, y_pred, dataset_name=""):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan}

    # 지표 계산
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    r2 = r2_score(y_true_clean, y_pred_clean)

    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

    # 로그 출력
    if dataset_name:
        log(f"\n   {dataset_name}:")
        log(f"      MAE:  ₩{mae:,.0f}/kg")
        log(f"      RMSE: ₩{rmse:,.0f}/kg")
        log(f"      MAPE: {mape:.2f}%")
        log(f"      R²:   {r2:.4f}")

    return metrics

def train_single_model(model_info):
    model_name, X_train, X_val, X_test, y_train, y_val, y_test, best_params, random_state = model_info

    print(f"\n{'=' * 70}")
    print(f"[PROCESS] {model_name} 학습 중...")
    print(f"{'=' * 70}")

    start = time.time()

    try:
        if model_name == 'LightGBM':
            model = LGBMRegressor(
                n_estimators=200,
                **best_params,
                n_jobs=1,
                random_state=random_state,
                verbose=-1
            )
        elif model_name == 'RandomForest':
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=random_state,
                verbose=0
            )
        elif model_name == 'XGBoost':
            model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_jobs=1,
                random_state=random_state,
                verbosity=0
            )
        else:
            raise ValueError(f"알 수 없는 모델: {model_name}")

        # 모델 학습
        model.fit(X_train, y_train)

        # 예측
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)

        # 평가 지표 계산
        mask = ~np.isnan(y_test)
        y_test_clean = y_test[mask]
        pred_test_clean = pred_test[mask]

        mae = mean_absolute_error(y_test_clean, pred_test_clean)
        rmse = np.sqrt(mean_squared_error(y_test_clean, pred_test_clean))
        mape = np.mean(np.abs((y_test_clean - pred_test_clean) / y_test_clean)) * 100
        r2 = r2_score(y_test_clean, pred_test_clean)

        metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

        train_time = time.time() - start
        print(f"{model_name} 완료 ({train_time:.2f}초) - Test MAPE: {mape:.2f}%")

        return (model_name, model, pred_train, pred_val, pred_test, metrics, None)

    except Exception as e:
        print(f"{model_name} 실패: {e}")
        import traceback
        traceback.print_exc()
        return (model_name, None, None, None, None, None, str(e))

# 모델 학습: LightGBM Enhanced, Random Forest, XGBoost
def train_base_models(X_train, X_val, X_test, y_train, y_val, y_test, random_state):
    log("\n" + "=" * 70)
    log("3개 베이스 모델 학습 (병렬 처리)")
    log("=" * 70)

    base_models = {}
    predictions = {}

    # 최적 하이퍼파라미터 (튜닝 결과)
    best_params = {
        'learning_rate': 0.1,
        'min_child_samples': 10,
        'num_leaves': 31,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'subsample': 0.7,
        'colsample_bytree': 0.8
    }

    # 멀티프로세싱 준비
    from multiprocessing import Pool, cpu_count

    tasks = [
        ('LightGBM', X_train, X_val, X_test, y_train, y_val, y_test, best_params, random_state),
        ('RandomForest', X_train, X_val, X_test, y_train, y_val, y_test, {}, random_state),
        ('XGBoost', X_train, X_val, X_test, y_train, y_val, y_test, {}, random_state)
    ]

    log(f"\n멀티프로세싱 사용 (3개 병렬 작업)")

    # 병렬 실행
    with Pool(processes=min(3, cpu_count())) as pool:
        results = pool.map(train_single_model, tasks)

    # 결과 수집
    for model_name, model, pred_train, pred_val, pred_test, metrics, error in results:
        if error is None:
            log(f"\n{model_name}:")
            log(f"   Test MAPE: {metrics['MAPE']:.2f}%")
            log(f"   Test MAE:  ₩{metrics['MAE']:,.0f}/kg")
            log(f"   Test R²:   {metrics['R2']:.4f}")

            base_models[model_name] = model
            predictions[model_name] = {
                'train': pred_train,
                'val': pred_val,
                'test': pred_test,
                'metrics': metrics
            }
        else:
            log(f"\n {model_name} 실패: {error}")

    return base_models, predictions

# 단순 평균 앙상블
def ensemble_simple_average(predictions, y_train, y_val, y_test):
    log("\n[앙상블 1] 단순 평균...")

    # 예측값 평균
    pred_train = np.mean([predictions['LightGBM']['train'],
                          predictions['RandomForest']['train'],
                          predictions['XGBoost']['train']], axis=0)
    pred_val = np.mean([predictions['LightGBM']['val'],
                        predictions['RandomForest']['val'],
                        predictions['XGBoost']['val']], axis=0)
    pred_test = np.mean([predictions['LightGBM']['test'],
                         predictions['RandomForest']['test'],
                         predictions['XGBoost']['test']], axis=0)

    # 평가 지표 계산
    train_metrics = calculate_metrics(y_train.values, pred_train, "Train")
    val_metrics = calculate_metrics(y_val.values, pred_val, "Val")
    test_metrics = calculate_metrics(y_test.values, pred_test, "Test")

    return {
        'name': 'Simple Average',
        'weights': [1 / 3, 1 / 3, 1 / 3],
        'predictions': {'train': pred_train, 'val': pred_val, 'test': pred_test},
        'metrics': {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
    }

# 가중 평균 앙상블
def ensemble_weighted_average(predictions, y_train, y_val, y_test):
    log("\n[앙상블 2] 가중 평균 (검증 세트 최적화)...")

    # 가중치 조합 탐색
    best_weights = None
    best_val_mape = float('inf')

    weight_combinations = []
    for w1 in np.arange(0.2, 0.7, 0.1):
        for w2 in np.arange(0.2, 0.7, 0.1):
            w3 = 1.0 - w1 - w2
            if 0.1 <= w3 <= 0.7:
                weight_combinations.append([w1, w2, w3])

    # 최적 가중치 찾기
    for weights in weight_combinations:
        pred_val = (weights[0] * predictions['LightGBM']['val'] +
                    weights[1] * predictions['RandomForest']['val'] +
                    weights[2] * predictions['XGBoost']['val'])

        val_mape = np.mean(np.abs((y_val.values - pred_val) / y_val.values)) * 100

        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_weights = weights

    log(f"\n   최적 가중치: LightGBM={best_weights[0]:.2f}, RF={best_weights[1]:.2f}, XGB={best_weights[2]:.2f}")

    # 최적 가중치 적용
    pred_train = (best_weights[0] * predictions['LightGBM']['train'] +
                  best_weights[1] * predictions['RandomForest']['train'] +
                  best_weights[2] * predictions['XGBoost']['train'])
    pred_val = (best_weights[0] * predictions['LightGBM']['val'] +
                best_weights[1] * predictions['RandomForest']['val'] +
                best_weights[2] * predictions['XGBoost']['val'])
    pred_test = (best_weights[0] * predictions['LightGBM']['test'] +
                 best_weights[1] * predictions['RandomForest']['test'] +
                 best_weights[2] * predictions['XGBoost']['test'])

    # 평가 지표 계산
    train_metrics = calculate_metrics(y_train.values, pred_train, "Train")
    val_metrics = calculate_metrics(y_val.values, pred_val, "Val")
    test_metrics = calculate_metrics(y_test.values, pred_test, "Test")

    return {
        'name': 'Weighted Average',
        'weights': best_weights,
        'predictions': {'train': pred_train, 'val': pred_val, 'test': pred_test},
        'metrics': {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
    }

# 스태킹 앙상블
def ensemble_stacking(predictions, y_train, y_val, y_test, model_dir, random_state):
    log("\n[앙상블 3] 스태킹 (Ridge 메타 모델)...")

    # 메타 피처 준비
    meta_train = np.column_stack([
        predictions['LightGBM']['train'],
        predictions['RandomForest']['train'],
        predictions['XGBoost']['train']
    ])
    meta_val = np.column_stack([
        predictions['LightGBM']['val'],
        predictions['RandomForest']['val'],
        predictions['XGBoost']['val']
    ])
    meta_test = np.column_stack([
        predictions['LightGBM']['test'],
        predictions['RandomForest']['test'],
        predictions['XGBoost']['test']
    ])

    # 메타 모델 학습
    meta_model = Ridge(alpha=1.0, random_state=random_state)
    meta_model.fit(meta_train, y_train)

    log(f"\n   메타 모델 계수:")
    log(f"      LightGBM: {meta_model.coef_[0]:.4f}")
    log(f"      RandomForest: {meta_model.coef_[1]:.4f}")
    log(f"      XGBoost: {meta_model.coef_[2]:.4f}")

    # 예측
    pred_train = meta_model.predict(meta_train)
    pred_val = meta_model.predict(meta_val)
    pred_test = meta_model.predict(meta_test)

    # 평가 지표 계산
    train_metrics = calculate_metrics(y_train.values, pred_train, "Train")
    val_metrics = calculate_metrics(y_val.values, pred_val, "Val")
    test_metrics = calculate_metrics(y_test.values, pred_test, "Test")

    # 메타 모델 저장
    meta_path = os.path.join(model_dir, 'stacking_meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_model, f)
    log(f"\n   메타 모델 저장: {meta_path}")

    return {
        'name': 'Stacking',
        'meta_model': meta_model,
        'predictions': {'train': pred_train, 'val': pred_val, 'test': pred_test},
        'metrics': {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
    }


# 비교
def create_comparison_dataframe(predictions, ensemble_results):
    log("\n" + "=" * 70)
    log("모델 성능 비교")
    log("=" * 70)

    comparison_data = []

    # 베이스 모델
    for model_name, result in predictions.items():
        comparison_data.append({
            'Model': model_name,
            'Type': 'Base',
            'Train_MAPE': result['metrics']['MAPE'],
            'Val_MAPE': result['metrics']['MAPE'],
            'Test_MAPE': result['metrics']['MAPE'],
            'Test_MAE': result['metrics']['MAE'],
            'Test_RMSE': result['metrics']['RMSE'],
            'Test_R2': result['metrics']['R2']
        })

    # 앙상블 모델
    for result in ensemble_results:
        comparison_data.append({
            'Model': result['name'],
            'Type': 'Ensemble',
            'Train_MAPE': result['metrics']['train']['MAPE'],
            'Val_MAPE': result['metrics']['val']['MAPE'],
            'Test_MAPE': result['metrics']['test']['MAPE'],
            'Test_MAE': result['metrics']['test']['MAE'],
            'Test_RMSE': result['metrics']['test']['RMSE'],
            'Test_R2': result['metrics']['test']['R2']
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_MAPE')

    log("\n전체 비교 (Test MAPE 기준 정렬):\n")
    log(comparison_df.to_string(index=False))

    best_model = comparison_df.iloc[0]['Model']
    best_mape = comparison_df.iloc[0]['Test_MAPE']

    log(f"\n최고 성능 모델: {best_model}")
    log(f"   Test MAPE: {best_mape:.2f}%")

    # LightGBM 대비 개선도 계산
    lgb_mape = comparison_df[comparison_df['Model'] == 'LightGBM']['Test_MAPE'].values[0]

    log(f"\nLightGBM Enhanced 대비 개선도:")
    for idx, row in comparison_df.iterrows():
        if row['Type'] == 'Ensemble':
            improvement = lgb_mape - row['Test_MAPE']
            improvement_pct = (improvement / lgb_mape) * 100 if lgb_mape > 0 else 0
            log(f"\n   {row['Model']}:")
            log(f"      MAPE: {lgb_mape:.2f}% → {row['Test_MAPE']:.2f}%")
            if improvement > 0:
                log(f"      개선: {improvement:.2f}%p ({improvement_pct:.1f}% 향상)")
            else:
                log(f"      변화: {improvement:.2f}%p ({improvement_pct:.1f}%)")

    return comparison_df