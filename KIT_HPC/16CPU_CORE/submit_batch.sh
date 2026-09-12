#!/bin/bash
# =========================================================
# 일괄 순차 처리(Batch Sequential) 제출 스크립트
# =========================================================

SCRIPT="run_batch.sh"
START_NUM=86761         # 시작 시드 번호
BATCH_SIZE=800         # 하나의 Job(qsub) 안에서 연속으로 돌릴 시뮬레이션 개수
CONCURRENT_JOBS=1     # 16코어 독점 (1개의 묶음만 던짐)

echo "========================================================="
echo "일괄 순차 처리(Batch Sequential) 방식 제출 시작"
echo "========================================================="

for ((i=0; i<CONCURRENT_JOBS; i++))
do
    # 체인 1: 101번부터 10개 (101~110)
    # 체인 2: 111번부터 10개 (111~120)
    CUR_START=$(( START_NUM + (i * BATCH_SIZE) ))
    
    echo "[Batch $((i+1))] 제출: Seed ${CUR_START} 부터 ${BATCH_SIZE}개"
    qsub -N "b${CUR_START}_batch" -v START_SEED=${CUR_START},BATCH_SIZE=${BATCH_SIZE} $SCRIPT
done

echo "---------------------------------------------------------"
echo "제출 완료! 각 Job이 ${BATCH_SIZE}개의 시드를 순서대로 돕니다."
