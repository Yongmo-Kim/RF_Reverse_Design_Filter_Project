#!/bin/bash
#PBS -N BPF_16C_BATCH
#PBS -q base_32
#PBS -l select=1:ncpus=16:mem=250gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /scratch/home/jungsu0910/Coral3/result/log/

# 1. 계정 및 가상환경 설정
cd /scratch/home/jungsu0910/Coral3
source /scratch/app/anaconda3/etc/profile.d/conda.sh
conda activate emerge1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMBA_NUM_THREADS=1

# 로그 폴더 생성
mkdir -p /scratch/home/jungsu0910/Coral3/result/log

# 변수 설정
BATCH_SIZE=${BATCH_SIZE:-10}
END_SEED=$(( START_SEED + BATCH_SIZE - 1 ))

echo "================================================="
echo "Batch Job Start: Seed $START_SEED to $END_SEED (16-Core Mode)"
echo "상세 내용은 시드별 .log 파일에 저장됩니다."
echo "================================================="

for (( SEED=START_SEED; SEED<=END_SEED; SEED++ ))
do
    INDIVIDUAL_LOG="/scratch/home/jungsu0910/Coral3/result/log/seed_${SEED}.log"
    
    echo "-------------------------------------------------"
    echo "[$(date)] Running Seed: $SEED (16-Core / Log: seed_${SEED}.log)"
    
    timeout 50m python simulation.py --seed $SEED > "$INDIVIDUAL_LOG" 2>&1
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 124 ]; then
        echo "=> [TimeOut] Seed $SEED 강제 종료 (50분 초과)"
    elif [ $EXIT_CODE -ne 0 ]; then
        echo "=> [Error] Seed $SEED 에러 발생 ($EXIT_CODE)"
    else
        echo "=> [Success] Seed $SEED 완료"
    fi
done

echo "================================================="
echo "Batch Job Finished!"
echo "================================================="
