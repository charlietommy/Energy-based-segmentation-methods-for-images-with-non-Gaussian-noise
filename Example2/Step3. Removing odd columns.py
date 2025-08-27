import pandas as pd

def remove_even_columns(input_file, output_file):
    # Read Excel file  
    df = pd.read_excel(input_file)
    
    # Get all column names  
    columns = df.columns.tolist()
    
    # Select columns to keep (first + odd)  
    # First column is index 0 - keep it  
    # Then keep original odd columns (1, 3, 5...)  
    columns_to_keep = [columns[0]]  # Keep the first column  
    for i in range(1, len(columns), 2):
        if i < len(columns):
            columns_to_keep.append(columns[i])
    
    # Keep only selected columns  
    filtered_df = df[columns_to_keep]
    
    # Save to new file  
    filtered_df.to_excel(output_file, index=False)
    
    print(f"处理完成，已保留以下列：{columns_to_keep}")
    return filtered_df

# Example usage  
if __name__ == "__main__":
    input_file = "C:/Users/charlietommy/Desktop/paper1picture_grayscale_histograms.xlsx"  # Input file path  
    output_file = "C:/Users/charlietommy/Desktop/grayscale_histograms.xlsx"  # Output file path  
    
    result_df = remove_even_columns(input_file, output_file)
    print(f"前5行结果预览:\n{result_df.head()}")