import kagglehub

path = kagglehub.dataset_download(
    "safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb"
)

print(path)