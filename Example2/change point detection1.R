library(readxl)

calculate_results <- function(X, P_X, k_range) {
  # Initialize empty DataFrame for results  
  result_data <- data.frame(k = integer(), result = numeric())
  
  # Ensure input data is numeric  
  X <- as.numeric(X)
  P_X <- as.numeric(P_X)
  
  # Handle possible NA values  
  X[is.na(X)] <- 0
  P_X[is.na(P_X)] <- 0
  
  # Compute expectation of X  
  E_X <- sum(X * P_X, na.rm = TRUE)
  
  # Loop through each k in k_range  
  for (k in k_range) {
    # Compute expectation of indicator function  
    E_I_X_lt_k <- sum(ifelse(X < k, 1, 0) * P_X, na.rm = TRUE)
    E_I_X_gt_k <- sum(ifelse(X >= k, 1, 0) * P_X, na.rm = TRUE)
    
    # Safely compute E_X_I_leq_k  
    if (any(X < k)) {
      E_X_I_leq_k <- sum(X[X < k] * P_X[X < k], na.rm = TRUE)
    } else {
      E_X_I_leq_k <- 0
    }
    
    # Compute covariance  
    cov_I_X_X <- E_X_I_leq_k - E_X * E_I_X_lt_k
    
    # Safely compute result, avoid division by zero  
    if (E_I_X_lt_k > 0 && E_I_X_gt_k > 0) {
      result <- (cov_I_X_X^2) / (E_I_X_lt_k * E_I_X_gt_k)
    } else {
      result <- 0
    }
    
    result_data <- rbind(result_data, data.frame(k = k, result = result))
    #cat("k =", k, ", Result:", result, "\n")
  }
  
  # Plot only if DataFrame is not empty  
  if (nrow(result_data) > 0) {
    plot(result_data$k, result_data$result, xlab=expression(theta), ylab=expression(Y(theta)))
    
    # Find max value and its index  
    kmax_index <- which.max(result_data$result)
    if (length(kmax_index) > 0) {  # Ensure max value is found  
      kmax <- result_data$k[kmax_index]
      abline(v = kmax, col = "red", lwd = 2)
      mtext(text = bquote(hat(tau)[2] == .(kmax)), 
            side = 1, at = kmax + 2, line = 0.3, col = "blue")
    }
  }
  
  # Return results DataFrame  
  return(kmax)
}

# Use tryCatch to handle errors  
tryCatch({
  # Read Excel file  
  file_path <- "/Users/yitingchen/Desktop/Research/Energy-based segmentation methods for non-Gaussian noised images/grayscale_histograms.xlsx"
  if (!file.exists(file_path)) {
    stop("File does not exist: ", file_path)
  }
  
  data <- read_excel(file_path)
  
  # Get column names  
  column_names <- colnames(data)
  print("Column names:")
  print(column_names)
  
  # Loop through each column  
  for (col_name in column_names) {
    cat("\nProcessing column:", col_name, "\n")
    
    # Get current column data  
    column_data <- data[[col_name]]
    
    # Call function and store result  
    results <- calculate_results(0:255, column_data, 1:255)
    
    # No need to print results (already printed inside)  
    # Optionally post-process results  
  }
}, error = function(e) {
  # Print error message  
  cat("Error:", conditionMessage(e), "\n")
})






#read grayscale histograms
grayhist=readxl::read_excel('C:/Users/charlietommy/Desktop/grayscale_histograms.xlsx')
dim(grayhist)
y=as.matrix(grayhist[,4])
calculate_results(0:255, y, 1:255)

otsu_result=c()
for (j in 1:1000){
  y=as.matrix(grayhist[,(j+1)])
  otsu_result[j]=calculate_results(0:255, y, 1:255)
}
otsu_result
write.csv(otsu_result, "C:/Users/charlietommy/Desktop/otsu.xlsx", row.names = FALSE)





