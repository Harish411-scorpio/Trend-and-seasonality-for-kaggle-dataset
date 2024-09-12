import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the SBI dataset (replace the path with the actual location of your file)
sbi_df = pd.read_csv('C:/Projects/Trend And Seasonality/SBI_data/SBI Dataset.csv', parse_dates=['Date'], index_col='Date')

# Let the user select the column to use ('Open', 'High', 'Low', or 'Close')
value_column = input("Enter the value type to analyze (Open, High, Low, Close): ")

# Ensure the input is valid
if value_column not in sbi_df.columns:
    raise ValueError(f"Invalid column: {value_column}. Please select from 'Open', 'High', 'Low', or 'Close'.")

# Define parameters
t = 20
d = 16
q = 8
X_t = 22.98886

# Initialize lists to store components
m_t = []
s_t = []
w_k = []

# Generate trend (m_t), seasonal (s_t), and weight (w_k)
for i in range(1, len(sbi_df) + 1):
    if i < len(sbi_df) // 2:
        # Upward trend with noise
        m_t_i = 10 + 0.8 * i + np.random.normal(0, 2)
    else:
        # Downward trend with noise
        m_t_i = (0.5 * X_t - q + X_t - q + 1 + X_t + q - 1 + 0.5 * X_t + q) / d
        m_t_i = 50 - 0.5 * (i - (len(sbi_df) // 2)) + np.random.normal(0, 2)

    # Sinusoidal seasonal component
    w_t_i = X_t - m_t_i
    s_t_i = w_t_i - (w_t_i / d)
    s_t_i = 10 * np.sin(2 * np.pi * i / 12)

    # Weight component (w_k) based on X_t
    w_k_i = ((2 * q - 1) ** -1) * X_t - i

    # Append the components to the respective lists
    m_t.append(m_t_i)
    s_t.append(s_t_i)
    w_k.append(w_k_i)

# Convert lists to numpy arrays
m_t = np.array(m_t)
s_t = np.array(s_t)
w_k = np.array(w_k)

# Calculate the result using the given components
result = w_k * (m_t + s_t)

# Calculate the residual component
residual_component = sbi_df[value_column] - result

# Plot the components
plt.figure(figsize=(12, 8))

# Plot original data
plt.subplot(4, 1, 1)
plt.plot(sbi_df.index, sbi_df[value_column], label=f'Original SBI {value_column} Value')
plt.legend(loc='upper left')

# Plot trend component
plt.subplot(4, 1, 2)
plt.plot(sbi_df.index, m_t, label='Trend (m_t)', color='orange')
plt.legend(loc='upper left')

# Plot seasonal component
plt.subplot(4, 1, 3)
plt.plot(sbi_df.index, s_t, label='Seasonal (s_t)', color='green')
plt.legend(loc='upper left')

# Plot residual component
plt.subplot(4, 1, 4)
plt.plot(sbi_df.index, residual_component, label='Residual', color='red')
plt.legend(loc='upper left')

# Final layout adjustments and show the plot
plt.tight_layout()
plt.show()
