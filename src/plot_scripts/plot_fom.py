import matplotlib.pyplot as plt

# Step 1: Read the loss values from the text file
with open('C:/Users/harry/Downloads/FOM_history_3-20250424T234639Z-001/FOM_history_3/4_14.txt', 'r') as file:
    loss_values = [float(line.strip()) for line in file if line.strip()]

# Step 2: Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(loss_values, marker='o', linestyle='-', color='b', label='FOM per iteration')
plt.xlabel('Iteration (Generation)')
plt.ylabel('Best FOM Value')
plt.title('Figure of Merit(FOM) Over Iterations')
plt.grid(True)
plt.legend()
plt.show()