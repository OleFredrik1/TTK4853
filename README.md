# TTK4853 - Experts in Teamwork - XAI

This is our XAI project for the course Experts in Teamwork at NTNU in 2021.
![Visualization sample](data/git_screen.png)  

Our brain tumor classification and visualization web application. The highlighted regions in the 3D model of the brain corresponds to the areas with the greatest activation in the VGG16 neural network. This activation is explained by the GradCAM++ algorithm. The 3D model of the brain is originally sized to 155 slices x 224 px x 244 px but is re-scaled to an eighth of that size in order to be feasibly rendered by the web app due to memory constraints. When running, the model can be freely panned, rotated and zoomed, and by hovering the mouse over an area in the model the user sees an information bar showing the value of the GradCAM++ activation.

### Installing dependencies  
```
pip3 install -r requirements.txt
```

### Running sample analysis  
```
chmod a+rx launch.sh
./launch.sh
```

### Using a custom brain volume  
```
python3 main.py --folder <foldername>
```
Note that the project is developed primarily for use on Linux/OSX based systems. 