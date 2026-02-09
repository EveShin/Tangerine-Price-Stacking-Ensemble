# 감귤 가격 예측 프로젝트 - 데이터 전처리

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
import time

warnings.filterwarnings('ignore')

# Config
DATA_DIR = r'C:\Users\shina\Desktop\IoT 인공지능\project\data\tan'
OUTPUT_DIR = r'C:\Users\shina\Desktop\IoT 인공지능\project'

# 기간
START_DATE = '2021-01-04'
END_DATE = '2025-11-24'

# Split 날짜
TRAIN_END = '2024-06-30'
VAL_END = '2024-12-31'

# 데이터 파일 경로
FILES = {
    'price': os.path.join(DATA_DIR, '가락시장_경락가격_감귤_농넷.xlsx'),
    'supply': os.path.join(DATA_DIR, '가락시장_반입량_감귤_농넷.xlsx'),
    'weather': os.path.join(DATA_DIR, 'krei_oasis_제주기상정보.csv'),
    'production': os.path.join(DATA_DIR, '공공데이터포털_제주특별자치도_품종별감귤생산현황_20241231.csv'),
    'consumption': os.path.join(DATA_DIR, '월별_대형마트_소비트렌드_감귤_농넷.csv'),
    'trade': os.path.join(DATA_DIR, '농산물_수출입_데이터_감귤_농넷2.xlsx')
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT = []

def log(message):
    REPORT.append(message)
    print(message)

def safe_read_file(filepath):
    try:
        if filepath.endswith('.csv'):
            try:
                return pd.read_csv(filepath, encoding='utf-8')
            except:
                return pd.read_csv(filepath, encoding='cp949')
        elif filepath.endswith('.xlsx'):
            return pd.read_excel(filepath)
    except Exception as e:
        log(f"파일 읽기 오류 {filepath}: {e}")
        return None


# 가격 데이터 전처리
def preprocess_price_data(df):
    log("\n" + "="*70)
    log("PHASE 1-1: 가격 데이터 전처리")
    log("="*70)
    start = time.time()

    df['date'] = pd.to_datetime(df['DATE'])
    original_count = len(df)

    # 가격 0 제거
    df = df[df['평균가격'] > 0].copy()
    zero_removed = original_count - len(df)
    log(f"0원 가격 제거: {zero_removed:,}건 ({zero_removed/original_count*100:.2f}%)")

    # kg당 가격 계산
    kg_map = {'3키로상자': 3, '5키로상자': 5, '10키로상자': 10}
    df['unit_kg'] = df['단위'].map(kg_map)
    df['price_per_kg'] = df['평균가격'] / df['unit_kg']

    # 이상치 처리 (IQR 방식)
    Q1 = df['price_per_kg'].quantile(0.25)
    Q3 = df['price_per_kg'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[df['price_per_kg'] > upper_bound]
    log(f"이상치 상한: {upper_bound:,.0f}원/kg")
    log(f"이상치 개수: {len(outliers):,}건 ({len(outliers)/len(df)*100:.2f}%)")

    df['price_per_kg'] = df['price_per_kg'].clip(upper=upper_bound)

    # 등급 인코딩
    grade_map = {'특': 4, '상': 3, '중': 2, '하': 1}
    df['grade_encoded'] = df['등급명'].map(grade_map)

    # 주간 집계
    df = df.set_index('date')
    price_weekly_avg = df.groupby(pd.Grouper(freq='W-MON')).agg({
        'price_per_kg': 'mean'
    }).reset_index()
    price_weekly_avg.columns = ['date', 'price_per_kg_avg']

    log(f"주간 집계 완료: {len(price_weekly_avg)}주")
    log(f"완료 시간: {time.time()-start:.2f}초")

    return price_weekly_avg

# 데이터 전처리
def preprocess_supply_data(df):
    log("\n" + "="*70)
    log("PHASE 1-2: 공급 데이터 전처리")
    log("="*70)
    start = time.time()

    df['date'] = pd.to_datetime(df['DATE'])
    market_cols = ['서울청과', '농협', '중앙청과', '동화청과', '한국청과']

    # 주간 집계
    df = df.set_index('date')
    supply_weekly = df.groupby(pd.Grouper(freq='W-MON')).agg({
        '총반입량': 'sum',
        '서울청과': 'sum',
        '농협': 'sum',
        '중앙청과': 'sum',
        '동화청과': 'sum',
        '한국청과': 'sum'
    }).reset_index()

    # 시장별 비율 계산
    for col in market_cols:
        supply_weekly[f'{col}_ratio'] = supply_weekly[col] / supply_weekly['총반입량']

    supply_weekly.columns = ['date', 'total_supply', 'supply_seoul', 'supply_nonghyup',
                             'supply_jungang', 'supply_donghwa', 'supply_hanguk',
                             'supply_seoul_ratio', 'supply_nonghyup_ratio',
                             'supply_jungang_ratio', 'supply_donghwa_ratio', 'supply_hanguk_ratio']

    log(f"주간 집계 완료: {len(supply_weekly)}주")
    log(f"완료 시간: {time.time()-start:.2f}초")

    return supply_weekly

# 기상 데이터 전처리
def preprocess_weather_data(df):
    log("\n" + "="*70)
    log("PHASE 1-3: 기상 데이터 전처리")
    log("="*70)
    start = time.time()

    # 센서 오류값 제거
    df = df[df['평균 기온(°C)'] != -99].copy()
    log(f"센서 오류값(-99도) 제거 완료")

    df['date'] = pd.to_datetime(df['조회일자'].astype(str), format='%Y%m%d')

    # 일별 평균
    weather_daily = df.groupby('date').agg({
        '평균 기온(°C)': 'mean',
        '최고 기온(°C)': 'mean',
        '최저 기온(°C)': 'mean',
        '평균 강수량(mm)': 'mean',
        '평균 일조시간(hr)': 'mean',
        '평균 순간최대풍속(m/s)': 'mean'
    }).reset_index()

    # 주간 집계
    weather_daily = weather_daily.set_index('date')
    weather_weekly = weather_daily.groupby(pd.Grouper(freq='W-MON')).agg({
        '평균 기온(°C)': 'mean',
        '최고 기온(°C)': 'mean',
        '최저 기온(°C)': 'mean',
        '평균 강수량(mm)': 'sum',
        '평균 일조시간(hr)': 'sum',
        '평균 순간최대풍속(m/s)': 'max'
    }).reset_index()

    weather_weekly.columns = ['date', 'temp_avg', 'temp_max', 'temp_min',
                              'rainfall', 'sunshine', 'wind_speed']

    # 기상 특성 변수
    weather_weekly['is_heavy_rain'] = (weather_weekly['rainfall'] > 50).astype(int)
    weather_weekly['is_strong_wind'] = (weather_weekly['wind_speed'] > 20).astype(int)
    weather_weekly['is_hot'] = (weather_weekly['temp_avg'] > 30).astype(int)
    weather_weekly['is_cold'] = (weather_weekly['temp_avg'] < 10).astype(int)

    log(f"주간 집계 완료: {len(weather_weekly)}주")
    log(f"완료 시간: {time.time()-start:.2f}초")

    return weather_weekly

# 생산량 데이터 전처리
def preprocess_production_data(df):
    log("\n" + "="*70)
    log("PHASE 1-4: 생산량 데이터 전처리")
    log("="*70)
    start = time.time()

    prod_cols = ['노지 온주밀감', '하우스 온주밀감', '월동 온주밀감', '노지 만감류', '하우스 만감류']

    df_production = df[df['구분'] == '생산량(톤)'].copy()
    df_production = df_production[['연도별'] + prod_cols]

    # 총 생산량 및 전년대비 증감률
    df_production['total_production'] = df_production[prod_cols].sum(axis=1)
    df_production['production_yoy'] = df_production['total_production'].pct_change()

    df_production.columns = ['year', 'production_noji', 'production_house',
                            'production_winter', 'production_noji_late',
                            'production_house_late', 'total_production', 'production_yoy']

    log(f"생산량 데이터: {len(df_production)}년 ({df_production['year'].min()}-{df_production['year'].max()})")
    log(f"완료 시간: {time.time()-start:.2f}초")

    return df_production

# 소비 데이터 전처리
def preprocess_consumption_data(df):
    log("\n" + "="*70)
    log("PHASE 1-5: 소비 데이터 전처리")
    log("="*70)
    start = time.time()

    try:
        df['년주차_str'] = df['년주차'].astype(str)
        df['year'] = df['년주차_str'].str[:4].astype(int)
        df['month'] = df['년주차_str'].str[4:6].astype(int)

        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')

        # 월별 집계
        consumption_monthly = df.groupby('date').agg({
            '추정판매량(개, 10개 기준)': 'mean',
            '평균가격(월10개)': 'mean'
        }).reset_index()

        consumption_monthly.columns = ['date', 'estimated_sales', 'avg_price_mart']

        log(f"소비 데이터: {len(consumption_monthly)}개월")
        log(f"완료 시간: {time.time()-start:.2f}초")

        return consumption_monthly

    except Exception as e:
        log(f"경고: 소비 데이터 파싱 실패 - {e}")
        log(f"소비 데이터 없이 진행")
        return None

# 수출입 데이터 전처리
def preprocess_trade_data(df):
    log("\n" + "="*70)
    log("PHASE 1-6: 수출입 데이터 전처리")
    log("="*70)
    start = time.time()

    df['year'] = df['DATE'] // 100
    df['month'] = df['DATE'] % 100
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')

    # 수출 데이터만 추출
    df_export = df[df['구분'] == '수출'].copy()
    trade_monthly = df_export.groupby('date').agg({
        '금액': 'sum',
        '중량': 'sum'
    }).reset_index()

    trade_monthly.columns = ['date', 'export_amount', 'export_weight']

    log(f"수출입 데이터: {len(trade_monthly)}개월")
    log(f"완료 시간: {time.time()-start:.2f}초")

    return trade_monthly

# 주간 베이스라인 생성
def create_weekly_baseline():
    log("\n" + "="*70)
    log("PHASE 2: 데이터 통합")
    log("="*70)

    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='W-MON')
    df_baseline = pd.DataFrame({'date': date_range})

    log(f"베이스라인 생성: {len(df_baseline)}주 ({START_DATE} ~ {END_DATE})")

    return df_baseline

# 데이터 병합
def merge_all_data(df_baseline, price_weekly, supply_weekly, weather_weekly,
                   production_yearly, consumption_monthly, trade_monthly):
    log("\n모든 데이터셋 병합 중...")
    start = time.time()

    df_final = df_baseline.copy()

    # 가격 데이터 병합
    df_final = df_final.merge(price_weekly, on='date', how='left')
    log(f"   가격 데이터 병합 완료")

    # 공급 데이터 병합
    df_final = df_final.merge(supply_weekly, on='date', how='left')
    log(f"   공급 데이터 병합 완료")

    # 기상 데이터 병합
    df_final = df_final.merge(weather_weekly, on='date', how='left')
    log(f"   기상 데이터 병합 완료")

    # 생산량 데이터 병합 (연도 기준)
    df_final['year'] = df_final['date'].dt.year
    df_final = df_final.merge(production_yearly, on='year', how='left')

    df_final['total_production'] = df_final['total_production'].fillna(method='ffill')
    df_final['production_yoy'] = df_final['production_yoy'].fillna(method='ffill')
    log(f"   생산량 데이터 병합 완료 (2024-2025 전방향 채우기)")

    # 소비 데이터 병합 (월 기준)
    if consumption_monthly is not None:
        df_final['year_month'] = df_final['date'].dt.to_period('M')
        consumption_monthly['year_month'] = consumption_monthly['date'].dt.to_period('M')
        consumption_monthly = consumption_monthly.drop('date', axis=1)

        df_final = df_final.merge(consumption_monthly, on='year_month', how='left')
        df_final = df_final.drop('year_month', axis=1)
        log(f"   소비 데이터 병합 완료 (NULL 유지)")
    else:
        log(f"   소비 데이터 건너뜀 (데이터 없음)")

    # 수출입 데이터 병합 (월 기준)
    df_final['year_month'] = df_final['date'].dt.to_period('M')
    trade_monthly['year_month'] = trade_monthly['date'].dt.to_period('M')
    trade_monthly = trade_monthly.drop('date', axis=1)

    df_final = df_final.merge(trade_monthly, on='year_month', how='left')
    df_final = df_final.drop('year_month', axis=1)
    log(f"   수출입 데이터 병합 완료 (NULL 유지)")

    log(f"\n데이터 통합 완료 ({time.time()-start:.2f}초)")
    log(f"   최종 크기: {df_final.shape[0]}행 x {df_final.shape[1]}열")

    return df_final

# 시간 특성 생성
def create_time_features(df):
    log("\n" + "="*70)
    log("PHASE 3: 피처 엔지니어링")
    log("="*70)
    start = time.time()

    log("\n시간 특성 생성 중...")

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear

    # 계절성 특성
    df['is_peak_season'] = df['month'].isin([10, 11, 12]).astype(int)
    df['is_off_season'] = df['month'].isin([5, 6, 7, 8]).astype(int)

    log(f"   시간 특성 7개 생성 완료")

    return df

# 시차 특성 생성
def create_lag_features(df):
    log("\n시차 특성 생성 중...")

    df['price_lag1w'] = df['price_per_kg_avg'].shift(1)
    df['price_lag4w'] = df['price_per_kg_avg'].shift(4)
    df['price_lag52w'] = df['price_per_kg_avg'].shift(52)

    df['supply_lag1w'] = df['total_supply'].shift(1)
    df['supply_lag4w'] = df['total_supply'].shift(4)

    log(f"   시차 특성 5개 생성 완료")

    return df

# 이동 평균 특성 생성
def create_rolling_features(df):
    log("\n이동 평균 특성 생성 중...")

    df['price_roll4w'] = df['price_per_kg_avg'].rolling(window=4, min_periods=1).mean()
    df['price_roll12w'] = df['price_per_kg_avg'].rolling(window=12, min_periods=1).mean()
    df['price_std4w'] = df['price_per_kg_avg'].rolling(window=4, min_periods=1).std()

    df['supply_roll4w'] = df['total_supply'].rolling(window=4, min_periods=1).mean()
    df['supply_std4w'] = df['total_supply'].rolling(window=4, min_periods=1).std()

    log(f"   이동 평균 특성 5개 생성 완료")

    return df

# 상호작용 특성 생성
def create_interaction_features(df):
    log("\n상호작용 특성 생성 중...")

    df['supply_x_month'] = df['total_supply'] * df['month']
    df['temp_x_month'] = df['temp_avg'] * df['month']

    log(f"   상호작용 특성 2개 생성 완료")

    return df

# 피처 엔지니어링
def apply_feature_engineering(df):
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_interaction_features(df)

    log(f"\n피처 엔지니어링 완료")

    return df

# data split
def split_data(df):
    log("\n" + "="*70)
    log("PHASE 4: TRAIN/VAL/TEST 분할")
    log("="*70)

    train = df[df['date'] <= TRAIN_END].copy()
    log(f"\nTrain: {len(train)}주 ({train['date'].min().date()} ~ {train['date'].max().date()})")

    val = df[(df['date'] > TRAIN_END) & (df['date'] <= VAL_END)].copy()
    log(f"Val:   {len(val)}주 ({val['date'].min().date()} ~ {val['date'].max().date()})")

    test = df[df['date'] > VAL_END].copy()
    log(f"Test:  {len(test)}주 ({test['date'].min().date()} ~ {test['date'].max().date()})")

    assert train['date'].max() < val['date'].min(), "Train/Val 데이터 누출!"
    assert val['date'].max() < test['date'].min(), "Val/Test 데이터 누출!"

    log(f"\n데이터 누출 없음 확인 완료")

    return train, val, test

# 품질 확인
def quality_check(df, name):
    log(f"\n품질 확인 - {name}:")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    log(f"   크기: {df.shape}")
    log(f"   결측치:")

    for col in missing[missing > 0].index:
        log(f"      {col}: {missing[col]}개 ({missing_pct[col]}%)")

    # 타겟 변수 확인
    if 'price_per_kg_avg' in df.columns:
        target_missing = df['price_per_kg_avg'].isnull().sum()
        target_missing_pct = target_missing / len(df) * 100

        log(f"\n   타겟 변수 (price_per_kg_avg):")
        log(f"      Non-null: {len(df) - target_missing}개 ({100-target_missing_pct:.2f}%)")
        log(f"      Missing: {target_missing}개 ({target_missing_pct:.2f}%)")

        if target_missing_pct > 20:
            log(f"      경고: 타겟 변수 결측률 높음!")

        if target_missing < len(df):
            log(f"      평균: {df['price_per_kg_avg'].mean():,.0f}원/kg")
            log(f"      표준편차: {df['price_per_kg_avg'].std():,.0f}원/kg")
            log(f"      최소값: {df['price_per_kg_avg'].min():,.0f}원/kg")
            log(f"      최대값: {df['price_per_kg_avg'].max():,.0f}원/kg")

# 데이터셋 저장
def save_datasets(train, val, test):
    log("\n" + "="*70)
    log("PHASE 5: 데이터셋 저장")
    log("="*70)

    train.to_csv(os.path.join(OUTPUT_DIR, 'train_data.csv'), index=False, encoding='utf-8-sig')
    log(f"저장 완료: train_data.csv ({len(train)}행)")

    val.to_csv(os.path.join(OUTPUT_DIR, 'val_data.csv'), index=False, encoding='utf-8-sig')
    log(f"저장 완료: val_data.csv ({len(val)}행)")

    test.to_csv(os.path.join(OUTPUT_DIR, 'test_data.csv'), index=False, encoding='utf-8-sig')
    log(f"저장 완료: test_data.csv ({len(test)}행)")

    # 피처 정보 저장
    feature_list = [col for col in train.columns if col != 'date']

    with open(os.path.join(OUTPUT_DIR, 'feature_info.txt'), 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("피처 정보\n")
        f.write("="*70 + "\n\n")
        f.write(f"총 피처 개수: {len(feature_list)}\n\n")

        f.write("피처 목록:\n")
        for i, feat in enumerate(feature_list, 1):
            f.write(f"{i:3d}. {feat}\n")

    log(f"저장 완료: feature_info.txt ({len(feature_list)}개 피처)")


def main():
    print("="*70)
    print("감귤 가격 예측 - 데이터 전처리 파이프라인")
    print("="*70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    pipeline_start = time.time()

    # 데이터 파일 로드
    log("\n" + "="*70)
    log("데이터 파일 로드")
    log("="*70)

    df_price = safe_read_file(FILES['price'])
    df_supply = safe_read_file(FILES['supply'])
    df_weather = safe_read_file(FILES['weather'])
    df_production = safe_read_file(FILES['production'])
    df_consumption = safe_read_file(FILES['consumption'])
    df_trade = safe_read_file(FILES['trade'])

    if df_price is None or df_supply is None or df_weather is None:
        log("필수 데이터 파일 누락!")
        return

    log("모든 데이터 파일 로드 완료")

    # 각 데이터 전처리
    price_weekly = preprocess_price_data(df_price)
    supply_weekly = preprocess_supply_data(df_supply)
    weather_weekly = preprocess_weather_data(df_weather)
    production_yearly = preprocess_production_data(df_production)
    consumption_monthly = preprocess_consumption_data(df_consumption)
    trade_monthly = preprocess_trade_data(df_trade)

    # 데이터 통합
    df_baseline = create_weekly_baseline()
    df_integrated = merge_all_data(df_baseline, price_weekly, supply_weekly,
                                    weather_weekly, production_yearly,
                                    consumption_monthly, trade_monthly)

    # 피처 엔지니어링
    df_features = apply_feature_engineering(df_integrated)

    # Train/Val/Test 분할
    train, val, test = split_data(df_features)

    # 품질 확인
    quality_check(train, "TRAIN")
    quality_check(val, "VALIDATION")
    quality_check(test, "TEST")

    # 저장
    save_datasets(train, val, test)

    # 완료
    pipeline_time = time.time() - pipeline_start

    log("\n" + "="*70)
    log("전처리 완료!")
    log("="*70)
    log(f"총 소요 시간: {pipeline_time:.2f}초 ({pipeline_time/60:.2f}분)")
    log(f"출력 디렉토리: {OUTPUT_DIR}")
    log("="*70)

    # 리포트 저장
    report_path = os.path.join(OUTPUT_DIR, 'preprocessing_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(REPORT))

    print(f"\n리포트 저장: {report_path}")

    print("\n" + "="*70)
    print("다음 단계:")
    print("="*70)
    print("1. preprocessing_report.txt에서 품질 확인")
    print("2. feature_info.txt에서 피처 확인")
    print("3. train_data.csv로 모델링 시작")
    print("="*70)


if __name__ == "__main__":
    main()