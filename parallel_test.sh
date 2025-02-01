#!/bin/bash

total_requests=5  # Default number of total_requests
while getopts "n:" opt; do
  case ${opt} in
    n ) total_requests=$OPTARG;;
    * ) echo "Usage: $0 [-n total_requests]"; exit 1;;
  esac
done
start_time=$(date +%s)  # Start time of the entire script

durations=()
for i in $(seq 1 $total_requests); do
  (
    req_start_time=$(date +%s)  # Capture start time in seconds
    curl --location --request POST 'http://158.160.158.166:8080/api/request' \
      --header 'Content-Type: application/json' \
      --data-raw '{
        "query": "В каком городе находится главный кампус Университета ИТМО?\n1. Москва\n2. Санкт-Петербург\n3. Екатеринбург\n4. Нижний Новгород",
        "id": 1
      }'
    req_end_time=$(date +%s)  # Capture end time in sec
    duration=$((req_end_time - req_start_time))
    durations+=($duration)
    echo -e "\nRequest $i took ${duration}s"
  ) &
done

wait  # Wait for all background processes to complete

end_time=$(date +%s)  # End time of the entire script
total_duration=$((end_time - start_time))

# # Calculate average and standard deviation
# sum=0
# for d in "${durations[@]}"; do
#   sum=$((sum + d))
# done
# average=$((sum / total_requests))

# sum_sq=0
# for d in "${durations[@]}"; do
#   diff=$((d - average))
#   sum_sq=$((sum_sq + diff * diff))
# done

# std_dev=$(echo "scale=2; sqrt($sum_sq / $total_requests)" | bc -l)
# echo "Average request time: ${average}s ± ${std_dev}s"

echo "All requests have completed. Total time: ${total_duration}s"
