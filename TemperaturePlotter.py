import numpy as np
import matplotlib.pyplot as plt

class TemperaturePlotter:
    def __init__(self, temps):
        """
        Initialize the TemperaturePlotter with temperature matrices.
        
        Parameters:
        temps (list of np.array): List of temperature matrices for each room.
        """
        self.temps = temps
        
    def plot(self):
        """ Plot temperature matrices for each room or all rooms in one plot. """
        plt.figure()
        num_rooms = len(self.temps)
        
        # Check for a single room or multiple rooms
        if num_rooms == 1:
            temp = self.temps[0]
            room_title = "Temperature in Room 1"
            num_cols = 1
        else:
            temp = np.full((self.temps[0].shape[0], num_rooms * self.temps[0].shape[1]), np.nan)
            for i, room_temp in enumerate(self.temps):
                temp[:, i * room_temp.shape[1]:(i + 1) * room_temp.shape[1]] = room_temp
            room_title = "Temperature in All Rooms"
            num_cols = num_rooms
            
        # Create meshgrid for plotting
        y_N, x_N = temp.shape
        X, Y = np.meshgrid(
            np.linspace(0, num_cols, x_N), np.linspace(0, 1.0, y_N)
        )
        
        # Contour plot
        plt.title(room_title)
        plt.contourf(X, Y, temp, levels=500, cmap=plt.cm.coolwarm, vmin=5, vmax=40)
        plt.colorbar(label='Temperature (°C)')
        plt.xlabel('Rooms (m)' if num_rooms > 1 else 'Width (m)')
        plt.ylabel('Height (m)')
        plt.show()



# We can plot it in Main like this:
    
#temperatures = []

#for i in range(iterations):
    # Solve for room 1
    #u1 = Method.solve(problem1)
    # Solve for room 2
    #u2 = Method.solve(problem2)
    # Solve for room 3
    #u3 = Method.solve(problem3)

    # store the temperature arrays after processing each iteration:
    #temperatures.append((u1, u2, u3))

# Now we can plot the temperatures from the last iteration:
#TemperaturePlotter(temperatures[-1])  # Using the last computed temperatures

#Or directly like this:
# plotter = TemperaturePlotter([u1, u2, u3])
# plotter.plot()
