raw_data = read.csv("./tibble_data.csv")
data_matrix = as.matrix(raw_data)
data_tibble = tibble::as.tibble(data_matrix)
outliers <- find_HDoutliers(data, k=1, knnsearchtype="brute", alpha=0.25)
display_HDoutliers(data, outliers) + ggplot2::ggtitle("Aimbot Anomaly Detection")

