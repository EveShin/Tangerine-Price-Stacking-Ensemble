"""
감귤 가격 예측 프로젝트
3개 모델 앙상블 메인 코드 (LightGBM Enhanced + Random Forest + XGBoost)
"""

import os
import time
from datetime import datetime
import network as naf

# ==================== 설정 ====================
DATA_DIR = r'C:\Users\shina\Desktop\IoT 인공지능\project'
OUTPUT_DIR = r'C:\Users\shina\Desktop\IoT 인공지능\project\final\ensemble_analysis'
MODEL_DIR = r'C:\Users\shina\Desktop\IoT 인공지능\project\final\ensemble_models'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
N_JOBS = 14

# ==================== 메인 함수 ====================
def main():
    naf.log("=" * 70)
    naf.log("앙상블 분석 - 3개 모델")
    naf.log("LightGBM Enhanced + Random Forest + XGBoost")
    naf.log("=" * 70)
    naf.log(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    naf.log("=" * 70)

    start_time = time.time()

    # 데이터 로드
    train, val, test = naf.load_data(DATA_DIR)
    X_train, X_val, X_test, y_train, y_val, y_test, features, train_df, val_df, test_df = naf.select_features(
        train, val, test
    )

    naf.log(f"\n피처 개수: {len(features)}")

    # 베이스 모델 학습
    base_models, predictions = naf.train_base_models(X_train, X_val, X_test, y_train, y_val, y_test, RANDOM_STATE)

    # 앙상블 방법 적용
    naf.log("\n" + "=" * 70)
    naf.log("앙상블 방법 적용")
    naf.log("=" * 70)

    ensemble_results = []

    # 방법 1: 단순 평균
    ensemble_results.append(naf.ensemble_simple_average(predictions, y_train, y_val, y_test))

    # 방법 2: 가중 평균
    ensemble_results.append(naf.ensemble_weighted_average(predictions, y_train, y_val, y_test))

    # 방법 3: 스태킹
    ensemble_results.append(naf.ensemble_stacking(predictions, y_train, y_val, y_test, MODEL_DIR, RANDOM_STATE))

    # 성능 비교
    comparison_df = naf.create_comparison_dataframe(predictions, ensemble_results)

    # 비교 결과 저장
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'ensemble_comparison_results.csv'), index=False)
    naf.log(f"\n비교 결과 저장: ensemble_comparison_results.csv")

    # 실행 시간 요약
    total_time = time.time() - start_time
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    naf.log("\n" + "=" * 70)
    naf.log("앙상블 분석 완료")
    naf.log("=" * 70)
    naf.log(f"\n총 실행 시간: {hours}시간 {minutes}분 {seconds}초")
    naf.log(f"결과 저장 위치: {OUTPUT_DIR}")

    # 리포트 저장
    report_path = os.path.join(OUTPUT_DIR, 'ENSEMBLE_REPORT.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(naf.REPORT))
    naf.log(f"\n리포트 저장: {report_path}")

    naf.log("\n" + "=" * 70)
    naf.log("🎉 모두 완료!")
    naf.log("=" * 70)


if __name__ == '__main__':
    main()